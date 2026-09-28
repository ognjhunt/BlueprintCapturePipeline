"""Promote a staged provider output to B2 before any cleanup may delete it.

A streamed policy canary leaves its provider archive in the staging store
(Spaces). Before anything deletes that object, ``promote_staged_provider_output``
copies it into the content-addressed B2 artifact store with a full readback,
indexes it, and writes a receipt the cleanup gate obeys
(``provider_output_promotion_records``). ``promote_then_cleanup`` runs the two
under one per-staging lock and then writes the staged-object absence proof.
Nothing here allocates or mutates a provider; object writes go only through the
classified configured-scene store's publishers.

Sources (review I2). Each staged key is judged by what is there now, never by
an assumption:

- the object the adapter observed (``observation`` = {size_bytes, etag}) must
  still be that object, else ``provider_output_remote_version_changed``;
- an SSH-recovered local ZIP (``local_archive``) is indexed and published by
  file, and removed only after its publication is verified (an offload behind
  its pointer, never a deletion of evidence). If the staged object also
  exists, it is hashed and recorded; different bytes are published too;
- without an observation, whatever is present is promoted;
- an absent object is recorded as ``absent_confirmed``.

Each promoted staged object's (size, ETag) is recorded as a version of its key,
so the gate deletes exactly the objects made durable. A later resume reuses a
receipt whose versions still match and promotes any object that does not.

Steps for the archive a consumer reads (the primary): one index pass
(``build_member_index``; limits 4 x ``maximum_archive_bytes`` expanded, 100,000
members, 2 GiB per member), one pinned copy into B2 with full readback
(``publish_configured_scene_stream``), then ``seal_durable_reference`` and the
index at ``<attempt>/provider_output_member_index.v1.json``. An archive the
index refuses is still promoted as evidence after one hash pass, and the
receipt carries ``provider_output_index_refused:<code>``. Transient transport
and publication failures get three attempts with backoff; identity refusals
get one.

Paired witness. A promoted, indexed output makes a staged witness redundant
only when every member of the witness (indexed from its own bytes) is in the
output: each file at ``cell_runs/00/<path>`` with the same SHA-256 and size,
and the witness manifest equal, as JSON, to the output's
``paired_witness_manifest.v1.json``. Otherwise -- including when the output is
absent or its index was refused -- the witness is promoted
(``policy-canary-paired-witness``); if the output's promotion failed, it is
deferred untouched.

B2 must be configured explicitly (review I4): all five
``BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_*_FILE`` variables. The artifact
client would otherwise fall back to the staging store's WAM credentials, so
promotion refuses first (``provider_output_promotion_artifact_store_not_configured``).

Receipt ``<staging>/provider_output_promotion.v1.json``: status
(``promoted``/``absent_confirmed``/``failed``), source, archive sha256 and size,
durable reference, member index {path, sha256, index_digest}, per-key
``staged_objects``, the witness disposition, attempts and blockers. No URL is
ever recorded.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .common import utc_now_iso
from .decision_evidence_contracts import canonical_digest
from .provider_output_member_index import (
    MAX_EXPANDED_BYTES,
    LocalArchiveRangeSource,
    ProviderOutputMemberIndexError,
    build_member_index,
    durable_reference_facts,
    read_indexed_member,
    seal_durable_reference,
)
from .provider_output_promotion_records import (
    ABSENCE_PROOF_FILENAME,
    DURABLE_STATES,
    RECEIPT_SCHEMA,
    ProviderOutputPromotionRecordError,
    key_sha256,
    load_promotion_receipt,
    load_staged_object_absence_proof,
    normalized_etag,
    staging_manifest_sha256,
    write_promotion_receipt,
    write_staged_object_absence_proof,
)
from .provider_output_range_transport import ProviderOutputRangeReader, ProviderOutputTransportError
from .native_task_arena_paired_witness_staging import SUFFIX as PAIRED_WITNESS_SUFFIX
from .task_evaluation_configured_scene_object_store import (
    _ARTIFACT_STORE_FILE_ENV,
    TaskEvaluationConfiguredSceneObjectStoreError,
    publish_configured_scene_artifact,
    publish_configured_scene_stream,
)
from .wam_provider_object_store import (
    SCHEMA_VERSION as STAGING_SCHEMA_VERSION,
    STAGING_MANIFEST_FILENAME,
    cleanup_staged_wam_provider_objects,
    presign_staged_object_get,
)

OUTPUT_ARTIFACT_KIND = "policy-canary-provider-output"
WITNESS_ARTIFACT_KIND = "policy-canary-paired-witness"
OUTPUT_FILENAME = "vast_provider_runtime_output.zip"
WITNESS_FILENAME = "policy_canary_paired_witness.zip"
INDEX_FILENAME = "provider_output_member_index.v1.json"
RESUME_FILENAME = "provider_output_resume.v1.json"
RESUME_SCHEMA = "provider_output_resume.v1"
LOCK_FILENAME = ".provider_output_promotion.lock"
WITNESS_CELL_PREFIX = "cell_runs/00/"
WITNESS_MANIFEST_MEMBER = "paired_witness_manifest.v1.json"
# The arena lane's attempt layout, which resume reads.
STAGING_DIRNAME = "object_store_staging"
PROVIDER_RUN_DIRNAME = "vast_provider_run"
COMMAND_RESULT_NAME = "vast_provider_command_result.json"
MAXIMUM_MEMBERS = 100_000
MAXIMUM_MEMBER_INFLATED_BYTES = 2 * 1024**3
MAXIMUM_MANIFEST_BYTES = 64 * 1024**2
DEFAULT_MAXIMUM_ARCHIVE_BYTES = 64 * 1024**3
PRESIGN_EXPIRATION_SECONDS = 2 * 3600
READER_DEADLINE_SECONDS = 2 * 3600
DEFAULT_LOCK_TIMEOUT_SECONDS = 3 * 3600
LOCK_POLL_SECONDS = 1.0
RETRY_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 2.0
_ARTIFACT_KIND = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")
_CODE = re.compile(r"[a-z0-9_]+(:[A-Za-z0-9_.-]+)?")
# Transport refusals the index pass reports under its own error type: these are
# not verdicts on the archive, so they fail (or retry) rather than promote.
TRANSPORT_CODES = frozenset({
    "provider_output_not_ready", "provider_output_remote_version_changed", "provider_output_http_failed",
    "provider_output_transport_failed", "provider_output_transfer_deadline_exceeded",
    "provider_output_range_truncated", "provider_output_range_overlong", "provider_output_archive_truncated",
    "provider_output_archive_overlong", "provider_output_content_length_invalid",
    "provider_output_content_range_invalid", "provider_output_range_unsupported",
    "provider_output_stream_status_invalid", "provider_output_content_encoding_invalid",
    "provider_output_immutable_version_missing", "provider_output_archive_size_invalid",
    "provider_output_read_memory_cap_exceeded", "provider_output_range_invalid",
    "provider_output_seek_invalid", "provider_output_transport_limits_invalid",
    # Same ETag, different directory bytes: an identity failure, not a structure verdict.
    "provider_output_archive_directory_changed",
})
TRANSIENT_CODES = frozenset({
    "provider_output_http_failed", "provider_output_transport_failed", "provider_output_range_truncated",
    "provider_output_archive_truncated", "provider_output_staged_presign_failed",
    "configured_scene_artifact_publication_failed", "configured_scene_artifact_readback_failed",
    "configured_scene_artifact_head_failed",
})


class ProviderOutputPromotionError(ValueError):
    """A typed promotion refusal; the message is the stable code."""


_TYPED_ERRORS = (ProviderOutputPromotionError, ProviderOutputMemberIndexError, ProviderOutputTransportError,
                 TaskEvaluationConfiguredSceneObjectStoreError)


def _code(exc: BaseException) -> str:
    """The stable code of a typed refusal; anything else is named only by its type."""
    code = str(exc) if isinstance(exc, _TYPED_ERRORS) else ""
    return code if _CODE.fullmatch(code) else f"provider_output_promotion_failed:{type(exc).__name__}"


def _retrying(step: Callable[[], Any], attempts: dict[str, int], role: str) -> Any:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        attempts[role] = attempt
        try:
            return step()
        except Exception as exc:  # noqa: BLE001 - every failure becomes one typed code
            code = _code(exc)
            if code.split(":", 1)[0] not in TRANSIENT_CODES or attempt == RETRY_ATTEMPTS:
                raise ProviderOutputPromotionError(code) from None
            time.sleep(RETRY_SLEEP_SECONDS * attempt)
    raise AssertionError("unreachable")


def _require_artifact_store() -> None:
    if not all(str(os.environ.get(name) or "").strip() for name in _ARTIFACT_STORE_FILE_ENV.values()):
        raise ProviderOutputPromotionError("provider_output_promotion_artifact_store_not_configured")


def _index_limits(maximum_archive_bytes: int) -> dict[str, int]:
    expanded = min(MAX_EXPANDED_BYTES, 4 * maximum_archive_bytes)
    return {"maximum_expanded_bytes": expanded, "maximum_members": MAXIMUM_MEMBERS,
            "maximum_member_inflated_bytes": min(MAXIMUM_MEMBER_INFLATED_BYTES, expanded)}


def _identity(reader) -> dict[str, Any]:
    return {"size_bytes": reader.identity["size_bytes"], "etag": reader.identity["etag"]}


def _same(version: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    return (version.get("size_bytes") == identity["size_bytes"]
            and normalized_etag(version.get("etag")) == normalized_etag(identity["etag"]))


def _observation(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if (not isinstance(value, Mapping) or type(value.get("size_bytes")) is not int or value["size_bytes"] < 1
            or not isinstance(value.get("etag"), str) or not normalized_etag(value["etag"])):
        raise ProviderOutputPromotionError("provider_output_promotion_observation_invalid")
    return {"size_bytes": value["size_bytes"], "etag": value["etag"]}


def _json_or_none(data: bytes) -> Any:
    try:
        return json.loads(data)
    except (UnicodeError, ValueError):
        return None


class _Promotion:
    """One promotion run: sources, the primary archive, and the per-key sections."""

    def __init__(self, *, staging_dir, attempt_root, artifact_kind, observation, local_archive,
                 maximum_archive_bytes, publisher, file_publisher, presign_staged, opener):
        self.staging = Path(staging_dir).expanduser().resolve()
        self.attempt = Path(attempt_root).expanduser().resolve()
        self.artifact_kind, self.maximum = artifact_kind, maximum_archive_bytes
        self.observation_argument, self.local_argument = observation, local_archive
        self.publisher, self.file_publisher = publisher, file_publisher
        self.presign_staged, self.opener = presign_staged, opener
        self.attempts: dict[str, int] = {}
        self.blockers: list[str] = []
        self.output_source = None
        self.output_index: dict | None = None
        self.output_reused = False
        self.output_succeeded = False
        self.local_removed_before = False
        self.local_verified: Path | None = None
        self.closers: list[Callable[[], None]] = []

    # -- inputs -----------------------------------------------------------
    def _staging(self) -> tuple[str, dict]:
        path = self.staging / STAGING_MANIFEST_FILENAME
        try:
            manifest = json.loads(path.read_text(encoding="utf-8")) if not path.is_symlink() else None
        except (OSError, UnicodeError, ValueError):
            manifest = None
        digest = staging_manifest_sha256(self.staging)
        output_key = str((manifest or {}).get("output_key") or "") if isinstance(manifest, dict) else ""
        if (not isinstance(manifest, dict) or digest is None
                or manifest.get("schema_version") != STAGING_SCHEMA_VERSION
                or manifest.get("status") not in {"completed", "blocked"} or not output_key):
            raise ProviderOutputPromotionError("provider_output_promotion_staging_manifest_invalid")
        self.output_key, self.witness_key, self.witness_maximum = output_key, None, self.maximum
        witness = manifest.get("paired_witness") if isinstance(manifest.get("paired_witness"), dict) else {}
        if witness.get("status") == "ready":
            if witness.get("witness_key") != output_key + PAIRED_WITNESS_SUFFIX:
                raise ProviderOutputPromotionError("provider_output_promotion_witness_key_mismatch")
            self.witness_key = witness["witness_key"]
            capacity = (witness.get("authority") or {}).get("maximum_archive_bytes")
            if type(capacity) is int and capacity > 0:
                self.witness_maximum = capacity
        return digest, manifest

    def _open(self, role: str, maximum: int):
        """A pinned reader on the staged object, or None when it is absent (HTTP 404)."""
        try:
            url = self.presign_staged(self.staging, object_role=role,
                                      expiration_seconds=PRESIGN_EXPIRATION_SECONDS)
            if not isinstance(url, str) or not url:
                raise ValueError("no url")
        except Exception:  # noqa: BLE001 - the URL or its error never enters a record
            raise ProviderOutputPromotionError("provider_output_staged_presign_failed") from None
        try:
            reader = ProviderOutputRangeReader(url, maximum_archive_bytes=maximum,
                                               deadline_seconds=READER_DEADLINE_SECONDS, opener=self.opener)
        except ProviderOutputTransportError as exc:
            if str(exc) == "provider_output_not_ready":
                return None
            raise ProviderOutputPromotionError(str(exc)) from None
        if not reader.identity.get("etag"):
            raise ProviderOutputPromotionError("provider_output_remote_etag_missing")
        return reader

    # -- publication ------------------------------------------------------
    def _verified(self, reference: Any, digest: str, size: int, mismatch: str) -> dict:
        try:
            facts = durable_reference_facts(reference)
        except ProviderOutputMemberIndexError:
            raise ProviderOutputPromotionError("provider_output_promotion_reference_invalid") from None
        if facts["digest"] != digest or facts["size_bytes"] != size:
            raise ProviderOutputPromotionError(mismatch)
        return facts

    def _publish_stream(self, reader, digest, size, filename, kind) -> dict:
        failure: dict[str, str] = {}

        def write_stream(sink):
            try:
                reader.stream_to(sink.write)
            except ProviderOutputTransportError as exc:
                failure.setdefault("code", str(exc))
                raise

        try:
            reference = self.publisher(write_stream=write_stream, digest=digest, size_bytes=size,
                                       filename=filename, artifact_kind=kind)
        except Exception as exc:  # noqa: BLE001 - the store wraps every failure
            raise ProviderOutputPromotionError(failure.get("code") or _code(exc)) from None
        return self._verified(reference, digest, size, "provider_output_promotion_reference_mismatch")

    def _hash(self, source) -> str:
        digest = hashlib.sha256()
        try:
            source.stream_to(digest.update)
        except ProviderOutputTransportError as exc:
            raise ProviderOutputPromotionError(str(exc)) from None
        return "sha256:" + digest.hexdigest()

    def _index(self, source, maximum: int) -> tuple[dict | None, str | None]:
        """Index ``source``; a structural refusal is returned, a transport one raised."""
        try:
            return build_member_index(source, **_index_limits(maximum)), None
        except ProviderOutputMemberIndexError as exc:
            if str(exc) in TRANSPORT_CODES:
                raise ProviderOutputPromotionError(str(exc)) from None
            return None, str(exc)

    def _write_index(self, sealed: dict) -> dict:
        path = self.attempt / INDEX_FILENAME
        if path.exists() or path.is_symlink():
            try:
                existing = json.loads(path.read_text(encoding="utf-8")) if not path.is_symlink() else None
            except (OSError, UnicodeError, ValueError):
                existing = None
            if existing != sealed:
                raise ProviderOutputPromotionError("provider_output_member_index_conflict")
        else:
            temporary = self.attempt / f".{INDEX_FILENAME}.{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("x", encoding="utf-8") as stream:
                    json.dump(sealed, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, path)
            except FileExistsError:
                raise ProviderOutputPromotionError("provider_output_member_index_conflict") from None
            finally:
                temporary.unlink(missing_ok=True)
        return {"path": INDEX_FILENAME, "index_digest": sealed["index_digest"],
                "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}

    # -- the output ---------------------------------------------------------
    def _primary_from_source(self, source, size: int, origin: str, publish: Callable[[str], dict]) -> dict:
        index, refusal = self._index(source, self.maximum)
        digest = index["archive"]["sha256"] if index is not None else self._hash(source)
        facts = publish(digest)
        member_index = None
        if index is not None:
            sealed = seal_durable_reference(index, facts)
            member_index = self._write_index(sealed)
            self.output_index = sealed
        self.output_source = source
        return {"source": origin, "archive_sha256": digest, "size_bytes": size, "durable_reference": facts,
                "member_index": member_index, "index_refusal": refusal}

    def _remote_primary(self, reader, origin: str) -> tuple[dict, dict]:
        identity = _identity(reader)
        primary = self._primary_from_source(
            reader, identity["size_bytes"], origin,
            lambda digest: self._publish_stream(reader, digest, identity["size_bytes"], OUTPUT_FILENAME,
                                                self.artifact_kind))
        return primary, {**identity, "archive_sha256": primary["archive_sha256"],
                         "durable_uri": primary["durable_reference"]["uri"]}

    def _local_primary(self, local: Path) -> dict:
        source = LocalArchiveRangeSource(local)
        self.closers.append(source.close)
        size = source.identity["size_bytes"]
        if size > self.maximum:
            raise ProviderOutputPromotionError("provider_output_archive_size_invalid")

        def publish(digest: str) -> dict:
            try:
                reference = self.file_publisher(path=local, artifact_kind=self.artifact_kind)
            except Exception as exc:  # noqa: BLE001 - the store wraps every failure
                raise ProviderOutputPromotionError(_code(exc)) from None
            return self._verified(reference, digest, size, "provider_output_local_archive_digest_mismatch")

        primary = self._primary_from_source(source, size, "ssh_local_zip", publish)
        self.local_verified = local
        return primary

    def _remote_version(self, reader, known: dict[str, str]) -> dict:
        """Make one more staged object durable: hash it; publish it unless those bytes already are."""
        identity = _identity(reader)
        digest = self._hash(reader)
        uri = known.get(digest)
        if uri is None:
            uri = self._publish_stream(reader, digest, identity["size_bytes"], OUTPUT_FILENAME,
                                       self.artifact_kind)["uri"]
        return {**identity, "archive_sha256": digest, "durable_uri": uri}

    def _output(self, prior: Mapping | None) -> tuple[dict | None, dict]:
        prior_durable = bool(prior and prior.get("status") == "promoted")
        primary = ({key: prior.get(key) for key in ("source", "archive_sha256", "size_bytes", "durable_reference",
                                                    "member_index", "index_refusal")}
                   if prior_durable else None)
        self.output_reused = primary is not None
        self.local_removed_before = bool(
            prior_durable and prior.get("local_copy_removed_after_verified_promotion") is True)
        versions = list(prior["staged_objects"]["output"]["versions"]) if prior_durable else []
        observation = self.observation_argument
        local = Path(self.local_argument) if self.local_argument else None
        local_present = bool(local and local.is_file() and not local.is_symlink())
        reader = self._open("output", self.maximum)
        if primary is None:
            if observation is not None and local_present:
                raise ProviderOutputPromotionError("provider_output_promotion_sources_ambiguous")
            if observation is not None:
                if reader is None:
                    raise ProviderOutputPromotionError("provider_output_observed_object_missing")
                if not _same(observation, _identity(reader)):
                    raise ProviderOutputPromotionError("provider_output_remote_version_changed")
                primary, version = self._remote_primary(reader, "remote_observation")
                versions.append(version)
            elif local_present:
                primary = self._local_primary(local)
                if reader is not None:
                    versions.append(self._remote_version(
                        reader, {primary["archive_sha256"]: primary["durable_reference"]["uri"]}))
            elif reader is not None:
                primary, version = self._remote_primary(reader, "remote_present")
                versions.append(version)
        else:
            known = {primary["archive_sha256"]: primary["durable_reference"]["uri"],
                     **{row["archive_sha256"]: row["durable_uri"] for row in versions if row.get("durable_uri")}}
            if local_present:
                if self._hash_local(local) not in known:
                    raise ProviderOutputPromotionError("provider_output_local_archive_differs_from_promoted")
                self.local_verified = local
            if reader is not None and not any(_same(row, _identity(reader)) for row in versions):
                versions.append(self._remote_version(reader, known))
        self.output_succeeded = True
        return primary, versions

    def _hash_local(self, local: Path) -> str:
        with LocalArchiveRangeSource(local) as source:
            return self._hash(source)

    # -- the witness --------------------------------------------------------
    def _witness_redundancy(self, reader, output_state: str) -> tuple[dict, str | None]:
        """Prove the witness adds nothing to the promoted output, or say why not.

        Returns the proof (or a not-proven record) and the witness digest when
        its index already computed it.
        """
        if output_state != "promoted":
            return {"status": "not_proven", "reason": "output_not_promoted"}, None
        if self.output_reused:
            return {"status": "not_proven", "reason": "promoted_output_not_read_in_this_run"}, None
        if self.output_index is None or self.output_source is None:
            return {"status": "not_proven", "reason": "promoted_output_has_no_member_index"}, None
        index, refusal = self._index(reader, self.witness_maximum)
        if index is None:
            return {"status": "not_proven", "reason": f"paired_witness_index_refused:{refusal}"}, None
        digest = index["archive"]["sha256"]
        output = {row["path"]: row for row in self.output_index["members"] if row["kind"] == "file"}
        rows = [row for row in index["members"] if row["kind"] == "file"]
        for row in rows:
            if row["path"] == WITNESS_MANIFEST_MEMBER:
                mine = output.get(WITNESS_MANIFEST_MEMBER)
                if mine is None:
                    return {"status": "not_proven", "reason": "paired_witness_manifest_absent_from_output"}, digest
                theirs = _json_or_none(self._read(reader, row))
                if theirs is None or theirs != _json_or_none(self._read(self.output_source, mine)):
                    return {"status": "not_proven", "reason": "paired_witness_manifest_differs"}, digest
                continue
            mine = output.get(WITNESS_CELL_PREFIX + row["path"])
            if mine is None or (mine["sha256"], mine["size"]) != (row["sha256"], row["size"]):
                return {"status": "not_proven", "reason": "paired_witness_member_absent_from_output"}, digest
        return {"status": "proven", "rule": "witness_rows_subset_of_output_cell_00_rows",
                "cell_prefix": WITNESS_CELL_PREFIX, "witness_member_count": len(rows),
                "witness_index_digest": index["index_digest"],
                "output_archive_sha256": self.output_index["archive"]["sha256"],
                "output_index_digest": self.output_index["index_digest"]}, digest

    @staticmethod
    def _read(source, row) -> bytes:
        try:
            return read_indexed_member(source, row, maximum_bytes=MAXIMUM_MANIFEST_BYTES)
        except ProviderOutputMemberIndexError as exc:
            raise ProviderOutputPromotionError(str(exc)) from None

    def _witness(self, prior: Mapping | None, output_state: str) -> dict:
        prior_section = (prior or {}).get("staged_objects", {}).get("paired_witness") if prior else None
        keep = prior_section is not None and prior_section.get("state") in DURABLE_STATES
        section = {"key_sha256": key_sha256(self.witness_key),
                   "state": prior_section["state"] if keep else "deferred",
                   "versions": list(prior_section["versions"]) if keep else []}
        if output_state == "failed":
            return section
        reader = self._open("paired_witness", self.witness_maximum)
        if reader is None:
            if not keep:
                section["state"] = "absent_confirmed"
            return section
        identity = _identity(reader)
        if any(_same(row, identity) for row in section["versions"]):
            return section
        redundancy, digest = self._witness_redundancy(reader, output_state)
        if redundancy["status"] == "proven":
            section["versions"].append({**identity, "archive_sha256": digest, "durable_uri": None,
                                        "redundancy": redundancy})
        else:
            digest = digest or self._hash(reader)
            uri = self._publish_stream(reader, digest, identity["size_bytes"], WITNESS_FILENAME,
                                       WITNESS_ARTIFACT_KIND)["uri"]
            section["versions"].append({**identity, "archive_sha256": digest, "durable_uri": uri,
                                        "redundancy": redundancy})
        section["state"] = ("promoted" if any(row.get("durable_uri") for row in section["versions"])
                            else "redundant_with_promoted_output")
        return section

    # -- the run ------------------------------------------------------------
    def run(self) -> dict:
        manifest_sha256 = output_key = None
        primary, output_section, witness_section, status = None, None, None, "failed"
        try:
            if (not isinstance(self.artifact_kind, str) or not _ARTIFACT_KIND.fullmatch(self.artifact_kind)
                    or type(self.maximum) is not int or not 0 < self.maximum <= MAX_EXPANDED_BYTES):
                raise ProviderOutputPromotionError("provider_output_promotion_arguments_invalid")
            self.observation_argument = _observation(self.observation_argument)
            manifest_sha256, _ = self._staging()
            output_key = self.output_key
            _require_artifact_store()
            prior = load_promotion_receipt(self.staging, staging_manifest_sha256=manifest_sha256)
            try:
                primary, versions = _retrying(lambda: self._output(prior), self.attempts, "output")
                status = "promoted" if primary else "absent_confirmed"
                output_section = {"key_sha256": key_sha256(output_key), "state": status, "versions": versions}
            except ProviderOutputPromotionError as exc:
                self.blockers.append(str(exc))
                primary = None
                prior_durable = bool(prior and prior.get("status") == "promoted")
                if prior_durable:  # what was made durable before stays durable
                    primary = {key: prior.get(key) for key in (
                        "source", "archive_sha256", "size_bytes", "durable_reference", "member_index",
                        "index_refusal")}
                    status = "promoted"
                output_section = {"key_sha256": key_sha256(output_key), "state": status,
                                  "versions": list(prior["staged_objects"]["output"]["versions"])
                                  if prior_durable else []}
            if primary and primary.get("index_refusal"):
                self.blockers.append(f"provider_output_index_refused:{primary['index_refusal']}")
            if self.witness_key is not None:
                try:
                    witness_section = _retrying(lambda: self._witness(prior, status), self.attempts,
                                                "paired_witness")
                except ProviderOutputPromotionError as exc:
                    self.blockers.append(f"paired_witness_promotion_failed:{exc}")
                    kept = ((prior or {}).get("staged_objects", {}).get("paired_witness") or {}) if prior else {}
                    durable = kept.get("state") in DURABLE_STATES
                    witness_section = {"key_sha256": key_sha256(self.witness_key),
                                       "state": kept["state"] if durable else "failed",
                                       "versions": list(kept["versions"]) if durable else []}
        except ProviderOutputPromotionError as exc:
            self.blockers.append(str(exc))
        finally:
            for close in self.closers:
                close()
        return self._receipt(status, manifest_sha256, output_key, primary, output_section, witness_section)

    def _receipt(self, status, manifest_sha256, output_key, primary, output_section, witness_section) -> dict:
        staged = {}
        if output_section is not None:
            staged["output"] = output_section
        if witness_section is not None:
            staged["paired_witness"] = witness_section
        latest = (witness_section or {}).get("versions") or [{}]
        witness = {"disposition": (witness_section or {}).get("state", "not_staged"),
                   "reference": latest[-1].get("durable_uri"), "redundancy": latest[-1].get("redundancy")}
        primary = primary or {}
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "generated_at": utc_now_iso(),
            "status": status,
            "attempt_root": str(self.attempt),
            "staging_manifest_sha256": manifest_sha256,
            "output_key_sha256": key_sha256(output_key) if output_key else None,
            "artifact_kind": self.artifact_kind,
            "maximum_archive_bytes": self.maximum,
            "observation": self.observation_argument if isinstance(self.observation_argument, dict) else None,
            "source": primary.get("source") or "none",
            "archive_sha256": primary.get("archive_sha256"),
            "size_bytes": primary.get("size_bytes"),
            "durable_reference": primary.get("durable_reference"),
            "member_index": primary.get("member_index"),
            "index_refusal": primary.get("index_refusal"),
            "local_copy_removed_after_verified_promotion": self.local_removed_before,
            "staged_objects": staged,
            "witness": witness,
            "attempts": dict(self.attempts),
            "blockers": sorted(set(self.blockers)),
            "private_url_recorded": False,
            "raw_secret_values_recorded": False,
        }
        if manifest_sha256 is None or not self.staging.is_dir():
            receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
            return receipt
        if self.local_verified is not None and status == "promoted" and self.output_succeeded:
            # The receipt names the durable copy before the local one goes, so a
            # crash in between leaves a pointer and a zip resume can verify.
            receipt["local_copy_removed_after_verified_promotion"] = True
            written = write_promotion_receipt(self.staging, receipt)
            try:
                self.local_verified.unlink()
            except OSError:
                receipt["local_copy_removed_after_verified_promotion"] = False
                receipt["blockers"] = sorted({*receipt["blockers"], "provider_output_local_copy_removal_failed"})
                written = write_promotion_receipt(self.staging, receipt)
            return written
        return write_promotion_receipt(self.staging, receipt)


def promote_staged_provider_output(
    *,
    staging_dir: str | Path,
    attempt_root: str | Path,
    artifact_kind: str = OUTPUT_ARTIFACT_KIND,
    observation: Mapping[str, Any] | None,
    local_archive: str | Path | None,
    maximum_archive_bytes: int,
    publisher: Callable[..., Mapping[str, Any]] = publish_configured_scene_stream,
    file_publisher: Callable[..., Mapping[str, Any]] = publish_configured_scene_artifact,
    presign_staged: Callable[..., str] = presign_staged_object_get,
    opener: Callable | None = None,
) -> dict:
    """Promote the staged output (and witness) and write the receipt; never raises.

    See the module docstring. Returns the receipt, written to the staging dir
    whenever its manifest could be read; a refusal is a ``failed`` receipt
    with one typed blocker.
    """
    try:
        return _Promotion(staging_dir=staging_dir, attempt_root=attempt_root, artifact_kind=artifact_kind,
                          observation=observation, local_archive=local_archive,
                          maximum_archive_bytes=maximum_archive_bytes, publisher=publisher,
                          file_publisher=file_publisher, presign_staged=presign_staged, opener=opener).run()
    except Exception as exc:  # noqa: BLE001 - promotion reports, it never raises
        return _unwritten_failure(staging_dir, artifact_kind, _code(exc))


def _unwritten_failure(staging_dir, artifact_kind, code: str) -> dict:
    receipt = {"schema_version": RECEIPT_SCHEMA, "generated_at": utc_now_iso(), "status": "failed",
               "staging_dir": str(staging_dir), "artifact_kind": artifact_kind, "staged_objects": {},
               "blockers": [code], "private_url_recorded": False, "raw_secret_values_recorded": False}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


@contextlib.contextmanager
def staging_lock(staging_dir: str | Path, *, timeout_seconds: float) -> Iterator[None]:
    """Hold the per-staging promotion lock (review I6), waiting up to ``timeout_seconds``.

    Promotion, its cleanup and resume take it, so two promoters of one
    staging dir run one after the other, and the second finds the first's
    receipt.
    """
    staging = Path(staging_dir)
    if staging.is_symlink() or not staging.is_dir():
        raise ProviderOutputPromotionError("provider_output_promotion_staging_missing")
    descriptor = os.open(staging / LOCK_FILENAME, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ProviderOutputPromotionError("provider_output_promotion_lock_timeout") from None
                time.sleep(LOCK_POLL_SECONDS)
        yield
    finally:
        os.close(descriptor)


def _cleanup(cleanup: Callable[[], Mapping[str, Any]]) -> dict:
    try:
        return dict(cleanup())
    except Exception as exc:  # noqa: BLE001 - a failed cleanup is evidence, never an escape
        return {"status": "blocked", "all_objects_absent": False, "objects": [],
                "blockers": [f"staged_object_cleanup_failed:{type(exc).__name__}"],
                "raw_secret_values_recorded": False}


def _write_proof(staging_dir, cleaned: Mapping, receipt: Mapping) -> None:
    if cleaned.get("status") != "completed" or cleaned.get("all_objects_absent") is not True:
        return
    with contextlib.suppress(ProviderOutputPromotionRecordError, OSError, ValueError):
        write_staged_object_absence_proof(staging_dir=staging_dir, cleanup=cleaned, promotion=receipt)


def promote_then_cleanup(*, cleanup: Callable[[], Mapping[str, Any]],
                         lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
                         **promotion: Any) -> tuple[dict, dict]:
    """Promote, then always run ``cleanup``, under the staging lock; never raises.

    ``promotion`` holds ``promote_staged_provider_output``'s arguments. When the
    cleanup proves every staged object absent, the staged-object absence proof
    is written (review C2). If the lock cannot be taken, promotion is skipped
    (a ``failed`` receipt, not written) and cleanup still runs: its gate needs
    no lock to stay safe.
    """
    staging = promotion.get("staging_dir")
    try:
        with staging_lock(Path(str(staging or "")), timeout_seconds=lock_timeout_seconds):
            receipt = promote_staged_provider_output(**promotion)
            cleaned = _cleanup(cleanup)
            _write_proof(staging, cleaned, receipt)
            return receipt, cleaned
    except ProviderOutputPromotionError as exc:
        return (_unwritten_failure(staging, promotion.get("artifact_kind", OUTPUT_ARTIFACT_KIND), str(exc)),
                _cleanup(cleanup))


def _recorded_observation(command_result: Path) -> dict | None:
    if command_result.is_symlink() or not command_result.is_file():
        return None
    try:
        value = json.loads(command_result.read_text(encoding="utf-8"))
        return _observation(value.get("provider_output_remote_observation"))
    except (OSError, UnicodeError, ValueError, AttributeError):
        return None


def resume_provider_output_promotion(
    attempt_root: str | Path,
    *,
    artifact_kind: str = OUTPUT_ARTIFACT_KIND,
    maximum_archive_bytes: int | None = None,
    cleanup: Callable[[], Mapping[str, Any]] | None = None,
    **dependencies: Any,
) -> dict:
    """Rerun promotion, the gated cleanup and the absence proof for one attempt.

    Reads the arena lane's layout: ``object_store_staging/``, the adapter's
    recorded ``provider_output_remote_observation``, and an SSH-recovered ZIP
    still in ``vast_provider_run/``. Writes ``provider_output_resume.v1.json``
    beside them and never rewrites the sealed lane result.
    """
    attempt = Path(attempt_root).expanduser().resolve()
    staging = attempt / STAGING_DIRNAME
    run = attempt / PROVIDER_RUN_DIRNAME
    local = run / OUTPUT_FILENAME
    receipt, cleaned = promote_then_cleanup(
        cleanup=cleanup or (lambda: cleanup_staged_wam_provider_objects(staging)),
        staging_dir=staging, attempt_root=attempt, artifact_kind=artifact_kind,
        observation=_recorded_observation(run / COMMAND_RESULT_NAME),
        local_archive=local if local.is_file() and not local.is_symlink() else None,
        maximum_archive_bytes=maximum_archive_bytes or DEFAULT_MAXIMUM_ARCHIVE_BYTES, **dependencies)
    blockers = list(receipt.get("blockers") or []) + list(cleaned.get("blockers") or [])
    try:
        proof = load_staged_object_absence_proof(staging)
    except ProviderOutputPromotionRecordError as exc:
        proof = None
        blockers.append(str(exc))
    if proof is None and not blockers:
        blockers.append("staged_object_absence_not_proven")
    result = {
        "schema_version": RESUME_SCHEMA,
        "generated_at": utc_now_iso(),
        "status": "completed" if proof is not None and not blockers else "blocked",
        "attempt_root": str(attempt),
        "promotion": {key: receipt.get(key) for key in ("status", "source", "receipt_digest", "blockers")},
        "cleanup": {key: cleaned.get(key) for key in ("status", "all_objects_absent", "blockers")},
        "absence_proof": ({"path": ABSENCE_PROOF_FILENAME, "proof_digest": proof["proof_digest"]}
                          if proof is not None else None),
        "lane_result_rewritten": False,
        "blockers": sorted(set(blockers)),
        "private_url_recorded": False,
        "raw_secret_values_recorded": False,
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    if attempt.is_dir():
        temporary = attempt / f".{RESUME_FILENAME}.{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, attempt / RESUME_FILENAME)
    return result


def main(argv: list[str] | None = None) -> int:
    """Resume one attempt's provider-output promotion, gated cleanup and absence proof.

    ``python -m blueprint_pipeline.provider_output_promotion resume --attempt-root <attempt>
    [--artifact-kind policy-canary-provider-output] [--maximum-archive-bytes N]``

    Run it as the ``blueprint`` service user with ``UMask=0077``, never as
    root (review I5). It writes the promotion receipt, the member index, the
    absence proof and ``provider_output_resume.v1.json`` into the attempt tree
    that the lane, the dispatcher and billing read later: files there owned by
    root would break those readers, and files readable by other users would
    expose run evidence. The operator door kind that launches it this way
    (``provider-output-resume``) comes with the lane wiring, as does ingestion
    on resume. It never rewrites the sealed lane result. Exit status 0 means
    the attempt's staged objects are proven absent with the output promoted or
    confirmed absent; 1 means the resume receipt names what is still blocked.
    """
    parser = argparse.ArgumentParser(description="Resume a streamed provider output's promotion.")
    commands = parser.add_subparsers(dest="command", required=True)
    resume = commands.add_parser("resume")
    resume.add_argument("--attempt-root", required=True, type=Path)
    resume.add_argument("--artifact-kind", default=OUTPUT_ARTIFACT_KIND)
    resume.add_argument("--maximum-archive-bytes", type=int, default=DEFAULT_MAXIMUM_ARCHIVE_BYTES)
    args = parser.parse_args(argv)
    result = resume_provider_output_promotion(args.attempt_root, artifact_kind=args.artifact_kind,
                                              maximum_archive_bytes=args.maximum_archive_bytes)
    print(json.dumps({key: result.get(key) for key in ("status", "blockers", "promotion", "cleanup",
                                                       "absence_proof")}, sort_keys=True))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover - module CLI
    sys.exit(main())
