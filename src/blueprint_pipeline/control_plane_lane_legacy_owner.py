"""Exact read-only target generation for historical owner attribution.

An old path-only census is never a registration or cleanup grant. The caller
must obtain a separate owner decision for the digest of this observation.
"""

from __future__ import annotations

import hashlib
import json
import fcntl
import os
import re
import stat
import time
from contextlib import ExitStack
from pathlib import Path

from .control_plane_disk_usage import allocated_bytes
from .decision_evidence_contracts import canonical_digest

_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
MAX_ENTRIES = 200_000
MAX_DEPTH = 64
MAX_SECONDS = 240.0


class LegacyOwnerError(ValueError):
    """Fixed code for unsafe or changed historical attribution evidence."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise LegacyOwnerError(code)


def _version(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            getattr(info, "st_blocks", 0))


def _directory_identity(info: os.stat_result) -> dict[str, int | str]:
    return dict(dev=info.st_dev, ino=info.st_ino, type="directory",
                mode=stat.S_IMODE(info.st_mode), uid=info.st_uid, gid=info.st_gid,
                ctime_ns=info.st_ctime_ns)


def _absolute(value: Path) -> None:
    _require(value.is_absolute() and value != Path("/") and ".." not in value.parts
             and len(value.parts) <= MAX_DEPTH and len(os.fsencode(value)) <= 4096,
             "legacy_target_unsafe")
    _require(all(part not in ("", ".", "..") and len(os.fsencode(part)) <= 255
                 for part in value.parts[1:]), "legacy_target_unsafe")


def _chain(path: Path, stack: ExitStack) -> list[tuple[int | None, str, int, tuple[int, ...]]]:
    """Keep every original named descriptor; never resolve a user path."""
    result: list[tuple[int | None, str, int, tuple[int, ...]]] = []
    previous: int | None = None
    for component in ("/", *path.parts[1:]):
        try:
            named = os.stat(component, dir_fd=previous, follow_symlinks=False)
            _require(stat.S_ISDIR(named.st_mode), "legacy_target_unsafe")
            descriptor = os.open(component, _DIRECTORY, dir_fd=previous)
            stack.callback(os.close, descriptor)
            opened = os.fstat(descriptor)
        except (OSError, ValueError) as error:
            if isinstance(error, LegacyOwnerError):
                raise
            raise LegacyOwnerError("legacy_target_unsafe") from None
        _require(_version(named) == _version(opened), "legacy_target_changed")
        result.append((previous, component, descriptor, _version(opened)))
        previous = descriptor
    return result


def _verify_chain(chain: list[tuple[int | None, str, int, tuple[int, ...]]]) -> None:
    for parent, name, descriptor, expected in chain:
        try:
            actual = os.fstat(descriptor)
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except OSError:
            raise LegacyOwnerError("legacy_target_changed") from None
        _require(_version(actual) == expected == _version(named), "legacy_target_changed")


def snapshot_generation(
    path: str | Path, *, allowed_roots: tuple[str | Path, ...],
    max_entries: int = MAX_ENTRIES, max_seconds: float = MAX_SECONDS,
) -> dict:
    """Bounded no-follow observation of an existing target, with no mutation.

    The tree digest includes names and inode metadata; it is an owner-label
    generation check and does not establish exclusive writer or reader lifetime.
    """
    target = Path(path)
    _absolute(target)
    _require(type(max_entries) is int and 0 < max_entries <= MAX_ENTRIES
             and type(max_seconds) in (int, float) and 0 < max_seconds <= MAX_SECONDS,
             "legacy_target_options_invalid")
    roots = tuple(Path(root) for root in allowed_roots)
    _require(0 < len(roots) <= 2, "legacy_target_options_invalid")
    for root in roots:
        _absolute(root)
    candidates = [root for root in roots if root in target.parents]
    _require(len(candidates) == 1, "legacy_target_unsafe")
    root = candidates[0]
    deadline = time.monotonic() + max_seconds
    entries: list[tuple[str, tuple[int, ...]]] = []
    bytes_allocated = 0

    def tick() -> None:
        _require(time.monotonic() < deadline, "legacy_target_measurement_incomplete")

    def walk(descriptor: int, relative: str, depth: int, device: int) -> None:
        nonlocal bytes_allocated
        tick()
        _require(depth <= MAX_DEPTH and len(entries) < max_entries,
                 "legacy_target_measurement_incomplete")
        start = os.fstat(descriptor)
        _require(stat.S_ISDIR(start.st_mode) and start.st_dev == device,
                 "legacy_target_measurement_incomplete")
        entries.append((relative, _version(start)))
        bytes_allocated += allocated_bytes(start)
        try:
            with os.scandir(descriptor) as iterator:
                names = sorted(entry.name for entry in iterator)
        except OSError:
            raise LegacyOwnerError("legacy_target_measurement_incomplete") from None
        _require(len(names) + len(entries) <= max_entries and len(names) == len(set(names)),
                 "legacy_target_measurement_incomplete")
        versions = {}
        for name in names:
            tick()
            _require(name not in ("", ".", "..") and len(os.fsencode(name)) <= 255,
                     "legacy_target_measurement_incomplete")
            if relative == "":
                _require(name != ".lane-scratch.v1.json", "legacy_target_existing_lease")
            try:
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError:
                raise LegacyOwnerError("legacy_target_measurement_incomplete") from None
            _require(info.st_dev == device and (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)),
                     "legacy_target_measurement_incomplete")
            versions[name] = _version(info)
            child_relative = f"{relative}/{name}" if relative else name
            if stat.S_ISDIR(info.st_mode):
                try:
                    child = os.open(name, _DIRECTORY, dir_fd=descriptor)
                    _require(_version(os.fstat(child)) == versions[name],
                             "legacy_target_changed")
                    walk(child, child_relative, depth + 1, device)
                except OSError:
                    raise LegacyOwnerError("legacy_target_measurement_incomplete") from None
                finally:
                    if "child" in locals():
                        os.close(child)
                        del child
            else:
                _require(len(entries) < max_entries, "legacy_target_measurement_incomplete")
                entries.append((child_relative, versions[name]))
                bytes_allocated += allocated_bytes(info)
        try:
            with os.scandir(descriptor) as iterator:
                _require(sorted(entry.name for entry in iterator) == names,
                         "legacy_target_changed")
            for name, version in versions.items():
                _require(_version(os.stat(name, dir_fd=descriptor, follow_symlinks=False)) == version,
                         "legacy_target_changed")
            _require(_version(os.fstat(descriptor)) == _version(start), "legacy_target_changed")
        except OSError:
            raise LegacyOwnerError("legacy_target_changed") from None

    with ExitStack() as stack:
        chain = _chain(target, stack)
        root_index = len(root.parts) - 1
        root_entry = chain[root_index]
        target_entry = chain[-1]
        _require(root_entry[3][0] == target_entry[3][0], "legacy_target_unsafe")
        walk(target_entry[2], "", 0, target_entry[3][0])
        _verify_chain(chain)
        tick()
        encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        return dict(path=str(target), root=str(root),
                    root_identity=_directory_identity(os.fstat(root_entry[2])),
                    target=_directory_identity(os.fstat(target_entry[2])),
                    tree=dict(digest="sha256:" + hashlib.sha256(encoded).hexdigest(),
                              entries=len(entries), allocated_bytes=bytes_allocated))


def build_version_packet(consent: dict, *, selected_path: str, generation: dict,
                         fresh_census: dict, now: float) -> dict:
    """Project a protected old consent and fresh survey into a reviewable packet.

    This output has no authority. A distinct owner decision must later bind its
    exact digest, and both generation and references must be freshly rechecked.
    """
    code = "legacy_owner_packet_incomplete"
    _require(type(now) in (int, float) and now >= 0 and now < float("inf"), code)
    _require(isinstance(consent, dict) and consent.get("schema_version") == "control_plane_lane_owner_consent.v1"
             and consent.get("execution_authorized") is False
             and consent.get("target_generation_bound") is False
             and isinstance(consent.get("decisions"), list)
             and isinstance(consent.get("expires_at_epoch"), (int, float))
             and now < consent["expires_at_epoch"], code)
    selected = [pair for pair in consent["decisions"]
                if isinstance(pair, dict) and isinstance(pair.get("census_row"), dict)
                and pair["census_row"].get("path") == selected_path]
    _require(len(selected) == 1, code)
    row, decision = selected[0]["census_row"], selected[0].get("decision")
    _require(isinstance(decision, dict) and decision.get("path") == selected_path
             and decision.get("action") == "register" and decision.get("cleanup") == "owner_review"
             and isinstance(decision.get("owner"), str) and decision["owner"]
             and type(decision.get("ttl_seconds")) is int and 0 < decision["ttl_seconds"] <= 1209600
             and row.get("references") == [] and row.get("unreadable") == 0, code)
    _require(isinstance(fresh_census, dict) and fresh_census.get("status") == "complete"
             and fresh_census.get("scan_errors") == [] and isinstance(fresh_census.get("rows"), list), code)
    fresh = [row for row in fresh_census["rows"]
             if isinstance(row, dict) and row.get("path") == selected_path]
    _require(len(fresh) == 1 and fresh[0].get("references") == []
             and fresh[0].get("unreadable") == 0, code)
    _require(isinstance(generation, dict) and generation.get("path") == selected_path
             and isinstance(generation.get("tree"), dict)
             and isinstance(generation["tree"].get("digest"), str)
             and generation.get("target", {}).get("type") == "directory", code)
    _require(isinstance(consent.get("census"), dict) and isinstance(consent.get("annotations"), dict)
             and isinstance(consent.get("consent_digest"), str)
             and isinstance(consent.get("policy_sha256"), str), code)
    packet = dict(schema_version="control_plane_lane_legacy_owner_packet.v1",
                  selected_path=selected_path, principal=consent["principal"], owner=decision["owner"],
                  old_consent=dict(consent_id=consent["consent_id"],
                                   sha256=consent["consent_digest"], target_generation_bound=False,
                                   census=consent["census"], annotations=consent["annotations"]),
                  policy_sha256=consent["policy_sha256"],
                  target_generation=generation,
                  fresh_reference_status="complete_no_observed_references",
                  observed_at_epoch=now,
                  expires_at_epoch=min(consent["expires_at_epoch"], now + decision["ttl_seconds"]),
                  cleanup="owner_review", execution_authorized=False, approval_required=True,
                  gc_eligible=False, references_clear=False, mutations=0)
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    _require(len(json.dumps(packet, sort_keys=True).encode()) <= 32768, code)
    return packet


def _approved_policy(packet: dict, policy_bytes: bytes, principal: str, owner: str,
                     *, now: float) -> dict:
    from . import control_plane_lane_owner_consents as owners
    from .control_plane_reference_budget import ReferenceCollectionBudget

    _require(isinstance(policy_bytes, bytes) and 0 < len(policy_bytes) <= owners.MAX_POLICY_BYTES,
             "legacy_owner_policy_changed")
    current = "sha256:" + hashlib.sha256(policy_bytes).hexdigest()
    _require(current == packet.get("policy_sha256"), "legacy_owner_policy_changed")
    _require(principal == packet.get("principal") and owner == packet.get("owner"),
             "legacy_owner_approval_principal_mismatch")
    budget = ReferenceCollectionBudget(monotonic=time.monotonic)
    try:
        try:
            policy = owners._policy(policy_bytes, principal, budget)
        except owners.OwnerCensusConsentError:
            raise LegacyOwnerError("legacy_owner_policy_changed") from None
        _require(owner in policy["owners"] and "register" in policy["allowed_actions"]
                 and now <= packet["observed_at_epoch"] + policy["max_consent_seconds"],
                 "legacy_owner_policy_changed")
        return policy
    finally:
        budget.close()


def approve_version_packet(packet: dict, *, ack_packet_digest: str,
                           current_policy_bytes: bytes, principal: str, owner: str,
                           now: float) -> dict:
    """Separate explicit owner decision after the generation packet is reviewable."""
    _require(isinstance(packet, dict)
             and packet.get("schema_version") == "control_plane_lane_legacy_owner_packet.v1"
             and packet.get("packet_digest") == canonical_digest(packet, digest_field="packet_digest")
             and packet.get("approval_required") is True
             and packet.get("execution_authorized") is False, "legacy_owner_packet_invalid")
    _require(ack_packet_digest == packet["packet_digest"], "legacy_owner_approval_ack_mismatch")
    _require(type(now) in (int, float) and packet["observed_at_epoch"] <= now < packet["expires_at_epoch"],
             "legacy_owner_approval_expired")
    policy = _approved_policy(packet, current_policy_bytes, principal, owner, now=now)
    decision = dict(schema_version="control_plane_lane_legacy_owner_approval.v1",
                    packet_digest=packet["packet_digest"], principal=principal, owner=owner,
                    approved_action="register_owner_review", approved_at_epoch=now,
                    expires_at_epoch=min(packet["expires_at_epoch"],
                                         now + policy["max_consent_seconds"]),
                    policy_sha256=packet["policy_sha256"], execution_authorized=False,
                    gc_eligible=False, mutations=0)
    decision["approval_digest"] = canonical_digest(decision, digest_field="approval_digest")
    return decision


def validate_registration(packet: dict, approval: dict, *, current_generation: dict,
                          fresh_census: dict, current_policy_bytes: bytes,
                          now: float) -> dict:
    """Check both decisions and current evidence; the result is owner_review only."""
    _require(isinstance(packet, dict) and isinstance(approval, dict)
             and packet.get("packet_digest") == canonical_digest(packet, digest_field="packet_digest")
             and approval.get("schema_version") == "control_plane_lane_legacy_owner_approval.v1"
             and approval.get("approval_digest") == canonical_digest(approval, digest_field="approval_digest")
             and approval.get("packet_digest") == packet["packet_digest"]
             and approval.get("approved_action") == "register_owner_review"
             and approval.get("execution_authorized") is False and approval.get("gc_eligible") is False,
             "legacy_owner_approval_invalid")
    _require(type(now) in (int, float) and now >= approval["approved_at_epoch"]
             and now < approval["expires_at_epoch"] and now < packet["expires_at_epoch"],
             "legacy_owner_approval_expired")
    _approved_policy(packet, current_policy_bytes, approval["principal"], approval["owner"], now=now)
    _require(current_generation == packet.get("target_generation"), "legacy_target_changed")
    _require(isinstance(fresh_census, dict) and fresh_census.get("status") == "complete"
             and fresh_census.get("scan_errors") == []
             and isinstance(fresh_census.get("rows"), list), "legacy_owner_references_incomplete")
    rows = [row for row in fresh_census["rows"] if isinstance(row, dict)
            and row.get("path") == packet.get("selected_path")]
    _require(len(rows) == 1 and rows[0].get("references") == [] and rows[0].get("unreadable") == 0,
             "legacy_owner_references_incomplete")
    result = dict(schema_version="control_plane_lane_legacy_owner_registration.v1",
                  path=packet["selected_path"], owner=approval["owner"],
                  principal=approval["principal"], packet_digest=packet["packet_digest"],
                  approval_digest=approval["approval_digest"],
                  target_generation=packet["target_generation"],
                  expires_at_epoch=approval["expires_at_epoch"], cleanup="owner_review",
                  classification="legacy_owner_review", gc_eligible=False,
                  references_clear=False, candidate_bytes=None, eta_seconds=None,
                  mutations=0)
    result["registration_digest"] = canonical_digest(result, digest_field="registration_digest")
    return result


class LegacyOwnerStore:
    """Root-owned append-only packet/decision/receipt chain for owner labels."""

    _ID = re.compile(r"[0-9a-f]{32}\Z")
    _ENTRY = re.compile(r"(?:[0-9a-f]{32}\.(?:packet|approval|registration|receipt)\.json|"
                        r"[0-9a-f]{64}\.[0-9a-f]{32}\.head\.json)\Z")
    _KINDS = frozenset(("packet", "approval", "registration", "receipt"))
    _CAP = 32768
    _MAX_RECORDS = 1024
    _MAX_BYTES = 64 * 1024 * 1024
    _LOCK = ".legacy-owner.lock"

    def __init__(self, files, root: str | Path):
        from . import control_plane_lane_owner_consents as owners

        self.files = files
        self.root = Path(root)
        try:
            self.parent, _ = files.parent(self.root / ".legacy-owner.probe", protected=True)
            owners._protected(os.fstat(self.parent), directory=True, mode=0o700)
            lock = files.open(self._LOCK, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                              parent=self.parent)
            info = os.fstat(lock)
            owners._protected(info, mode=0o600)
            _require(info.st_size == 0, "legacy_owner_store_unsafe")
            files.records.append(owners._Acquired(lock, self.parent, self._LOCK, info))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._scan()
        except (OSError, owners.OwnerCensusConsentError) as error:
            raise LegacyOwnerError("legacy_owner_store_unsafe") from error

    def _scan(self) -> list[str]:
        from . import control_plane_lane_owner_consents as owners

        count, size, names = 0, 0, []
        try:
            owners._protected(os.fstat(self.parent), directory=True, mode=0o700)
            with os.scandir(self.parent) as iterator:
                for entry in iterator:
                    self.files.budget.charge("entries")
                    count += 1
                    _require(count <= self._MAX_RECORDS + 1, "legacy_owner_store_full")
                    info = os.stat(entry.name, dir_fd=self.parent, follow_symlinks=False)
                    owners._protected(info, mode=0o600)
                    _require(entry.name == self._LOCK or self._ENTRY.fullmatch(entry.name) is not None,
                             "legacy_owner_store_unsafe")
                    _require(0 <= info.st_size <= self._CAP, "legacy_owner_store_unsafe")
                    if entry.name == self._LOCK:
                        _require(info.st_size == 0, "legacy_owner_store_unsafe")
                    else:
                        names.append(entry.name)
                    size += info.st_size
                    _require(size <= self._MAX_BYTES, "legacy_owner_store_full")
            self.files.verify()
            return sorted(names)
        except (OSError, owners.OwnerCensusConsentError) as error:
            raise LegacyOwnerError("legacy_owner_store_unsafe") from error

    @staticmethod
    def _payload(record: dict) -> bytes:
        _require(type(record) is dict, "legacy_owner_record_invalid")
        try:
            raw = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise LegacyOwnerError("legacy_owner_record_invalid") from None
        _require(0 < len(raw) <= LegacyOwnerStore._CAP, "legacy_owner_record_invalid")
        return raw

    def _record_path(self, name: str) -> Path:
        _require(self._ENTRY.fullmatch(name) is not None, "legacy_owner_record_invalid")
        return self.root / name

    def _read(self, name: str) -> dict:
        from . import control_plane_lane_owner_consents as owners
        from . import control_plane_lane_scratch_decisions as retained

        try:
            raw, _ = self.files.read(self._record_path(name), cap=self._CAP,
                                     protected=True, mode=0o600)
            value = retained._document(raw, self._CAP, _work_budget=self.files.budget)
            _require(type(value) is dict and raw == self._payload(value),
                     "legacy_owner_record_invalid")
            self.files.verify()
            return value
        except (OSError, owners.OwnerCensusConsentError, retained.CensusDecisionError) as error:
            raise LegacyOwnerError("legacy_owner_record_invalid") from error

    def read(self, packet_id: str, kind: str) -> dict:
        _require(isinstance(packet_id, str) and self._ID.fullmatch(packet_id) is not None
                 and kind in self._KINDS, "legacy_owner_record_invalid")
        return self._read(f"{packet_id}.{kind}.json")

    def _publish_name(self, name: str, record: dict) -> None:
        from . import control_plane_lane_owner_consents as owners

        payload = self._payload(record)
        _require(len(self._scan()) < self._MAX_RECORDS, "legacy_owner_store_full")
        try:
            current = os.stat(name, dir_fd=self.parent, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        except OSError:
            raise LegacyOwnerError("legacy_owner_record_conflict") from None
        if current is not None:
            _require(stat.S_ISREG(current.st_mode) and current.st_nlink == 1,
                     "legacy_owner_record_conflict")
            try:
                existing = self._read(name)
            except LegacyOwnerError:
                raise LegacyOwnerError("legacy_owner_record_conflict") from None
            _require(existing == record, "legacy_owner_record_conflict")
            return
        try:
            self.files.verify()
            owners._publish(self.files, self.parent, name, payload, mode=0o600, immutable=True)
        except (OSError, owners.OwnerCensusConsentError) as error:
            raise LegacyOwnerError("legacy_owner_record_conflict") from error
        _require(self._read(name) == record, "legacy_owner_record_conflict")

    def publish(self, packet_id: str, kind: str, record: dict) -> None:
        _require(isinstance(packet_id, str) and self._ID.fullmatch(packet_id) is not None
                 and kind in self._KINDS, "legacy_owner_record_invalid")
        self._publish_name(f"{packet_id}.{kind}.json", record)

    @staticmethod
    def _head_name(path: str, packet_id: str) -> str:
        _require(isinstance(path, str) and path.startswith("/") and ".." not in Path(path).parts
                 and isinstance(packet_id, str) and LegacyOwnerStore._ID.fullmatch(packet_id) is not None,
                 "legacy_owner_record_invalid")
        return hashlib.sha256(path.encode("utf-8")).hexdigest() + "." + packet_id + ".head.json"

    def publish_head(self, path: str, packet_id: str, registration: dict) -> None:
        _require(registration.get("path") == path, "legacy_owner_record_invalid")
        head = dict(schema_version="control_plane_lane_legacy_owner_head.v1",
                    packet_id=packet_id, path=path,
                    registration_digest=canonical_digest(registration),
                    gc_eligible=False, references_clear=False, mutations=0)
        self._publish_name(self._head_name(path, packet_id), head)

    def committed_heads(self) -> list[dict]:
        heads = []
        for name in self._scan():
            if not name.endswith(".head.json"):
                continue
            head = self._read(name)
            packet_id, path = head.get("packet_id"), head.get("path")
            _require(isinstance(packet_id, str) and isinstance(path, str)
                     and name == self._head_name(path, packet_id)
                     and head.get("schema_version") == "control_plane_lane_legacy_owner_head.v1"
                     and head.get("gc_eligible") is False
                     and head.get("references_clear") is False, "legacy_owner_record_invalid")
            registration = self.read(packet_id, "registration")
            receipt = self.read(packet_id, "receipt")
            _require(registration.get("path") == path and receipt.get("registration") == registration
                     and head.get("registration_digest") == canonical_digest(registration),
                     "legacy_owner_record_invalid")
            heads.append(head)
        return heads
