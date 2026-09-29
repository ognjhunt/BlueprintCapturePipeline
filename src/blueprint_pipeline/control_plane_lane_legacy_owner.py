"""Exact read-only target generation for historical owner attribution.

An old path-only census is never a registration or cleanup grant. The caller
must obtain a separate owner decision for the digest of this observation.
"""

from __future__ import annotations

import hashlib
import json
import fcntl
import math
import os
import re
import secrets
import stat
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .control_plane_disk_usage import allocated_bytes
from .decision_evidence_contracts import canonical_digest

_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
MAX_ENTRIES = 200_000
MAX_DEPTH = 64
MAX_SECONDS = 240.0
MAX_GENERATION_BYTES = 8 * 1024 * 1024


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


def _root_owned_publication(info: os.stat_result) -> bool:
    return (stat.S_ISREG(info.st_mode) and info.st_uid == info.st_gid == 0
            and stat.S_IMODE(info.st_mode) == 0o600)


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
    digest = hashlib.sha256()
    digest.update(b"[")
    entry_count = 0
    encoded_bytes = 2  # Opening and closing JSON array brackets.
    name_bytes = 0
    bytes_allocated = 0

    def tick() -> None:
        _require(time.monotonic() < deadline, "legacy_target_measurement_incomplete")

    def add_entry(relative: str, version: tuple[int, ...]) -> None:
        nonlocal entry_count, encoded_bytes
        fragment = json.dumps((relative, version), separators=(",", ":")).encode()
        additional = len(fragment) + (1 if entry_count else 0)
        _require(entry_count < max_entries
                 and encoded_bytes + additional <= MAX_GENERATION_BYTES,
                 "legacy_target_measurement_incomplete")
        if entry_count:
            digest.update(b",")
        digest.update(fragment)
        entry_count += 1
        encoded_bytes += additional

    def walk(descriptor: int, relative: str, depth: int, device: int) -> None:
        nonlocal bytes_allocated, name_bytes
        tick()
        _require(depth <= MAX_DEPTH and entry_count < max_entries,
                 "legacy_target_measurement_incomplete")
        start = os.fstat(descriptor)
        _require(stat.S_ISDIR(start.st_mode) and start.st_dev == device,
                 "legacy_target_measurement_incomplete")
        add_entry(relative, _version(start))
        bytes_allocated += allocated_bytes(start)
        try:
            with os.scandir(descriptor) as iterator:
                names = []
                for entry in iterator:
                    tick()
                    name_bytes += len(os.fsencode(entry.name))
                    _require(name_bytes <= MAX_GENERATION_BYTES
                             and len(names) + entry_count < max_entries,
                             "legacy_target_measurement_incomplete")
                    names.append(entry.name)
                names.sort()
        except OSError:
            raise LegacyOwnerError("legacy_target_measurement_incomplete") from None
        _require(len(names) + entry_count <= max_entries and len(names) == len(set(names)),
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
                add_entry(child_relative, versions[name])
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
        digest.update(b"]")
        relative = target.relative_to(root).parts
        # Bind every named ancestor as well as the target. A directory moved
        # between lanes must not inherit the previous owner's packet.
        ancestors = [
            dict(path=str(root.joinpath(*relative[:index])),
                 identity=_directory_identity(os.fstat(chain[root_index + index][2])))
            for index in range(1, len(relative))
        ]
        lane_root = ancestors[0] if relative[:1] == ("lanes",) and ancestors else None
        lane = ancestors[1] if relative[:1] == ("lanes",) and len(ancestors) > 1 else None
        return dict(path=str(target), root=str(root),
                    root_identity=_directory_identity(os.fstat(root_entry[2])),
                    ancestors=ancestors, lane_root=lane_root, lane=lane,
                    target=_directory_identity(os.fstat(target_entry[2])),
                    tree=dict(digest="sha256:" + digest.hexdigest(),
                              entries=entry_count, allocated_bytes=bytes_allocated))


def _reference_status(census: dict, selected_path: str, code: str) -> tuple[str, dict]:
    _require(isinstance(census, dict) and isinstance(census.get("rows"), list), code)
    errors = census.get("scan_errors")
    if census.get("status") == "complete" and errors == []:
        mode = "complete"
    elif (census.get("status") == "incomplete"
          and errors == ["process_inventory_unreadable"]
          and type(census.get("candidate_count")) is int
          and census["candidate_count"] == len(census["rows"])):
        mode = "unknown"
    else:
        raise LegacyOwnerError(code)
    rows = [row for row in census["rows"] if isinstance(row, dict)
            and row.get("path") == selected_path]
    _require(len(rows) == 1 and rows[0].get("references") == []
             and rows[0].get("unreadable") == 0, code)
    return mode, rows[0]


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
    process_mode, fresh = _reference_status(fresh_census, selected_path, code)
    _require(isinstance(generation, dict) and generation.get("path") == selected_path
             and isinstance(generation.get("tree"), dict)
             and isinstance(generation["tree"].get("digest"), str)
             and generation.get("target", {}).get("type") == "directory", code)
    _require(isinstance(consent.get("census"), dict) and isinstance(consent.get("annotations"), dict)
             and isinstance(consent.get("consent_digest"), str)
             and isinstance(consent.get("policy_sha256"), str)
             and isinstance(consent.get("_raw_record_sha256"), str)
             and re.fullmatch(r"sha256:[0-9a-f]{64}", consent["_raw_record_sha256"]) is not None
             and type(consent.get("_raw_record_size_bytes")) is int
             and 0 < consent["_raw_record_size_bytes"] <= 512 * 1024, code)
    packet = dict(schema_version="control_plane_lane_legacy_owner_packet.v1",
                  selected_path=selected_path, principal=consent["principal"], owner=decision["owner"],
                  old_consent=dict(consent_id=consent["consent_id"],
                                   sha256=consent["consent_digest"], target_generation_bound=False,
                                   record_sha256=consent["_raw_record_sha256"],
                                   record_size_bytes=consent["_raw_record_size_bytes"],
                                   census=consent["census"], annotations=consent["annotations"]),
                  policy_sha256=consent["policy_sha256"],
                  target_generation=generation,
                  fresh_census_digest=canonical_digest(fresh_census),
                  fresh_selected_row=dict(fresh),
                  fresh_reference_status=("complete_no_observed_references" if process_mode == "complete"
                                          else "process_fd_references_unknown"),
                  process_fd_references=process_mode,
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
                           now: float, ack_process_fd_unknown: bool = False) -> dict:
    """Separate explicit owner decision after the generation packet is reviewable."""
    _require(isinstance(packet, dict)
             and packet.get("schema_version") == "control_plane_lane_legacy_owner_packet.v1"
             and packet.get("packet_digest") == canonical_digest(packet, digest_field="packet_digest")
             and packet.get("approval_required") is True
             and packet.get("execution_authorized") is False, "legacy_owner_packet_invalid")
    _require(ack_packet_digest == packet["packet_digest"], "legacy_owner_approval_ack_mismatch")
    _require(type(ack_process_fd_unknown) is bool
             and packet.get("process_fd_references") in ("complete", "unknown")
             and ack_process_fd_unknown == (packet["process_fd_references"] == "unknown"),
             "legacy_owner_approval_ack_mismatch")
    _require(type(now) in (int, float) and packet["observed_at_epoch"] <= now < packet["expires_at_epoch"],
             "legacy_owner_approval_expired")
    policy = _approved_policy(packet, current_policy_bytes, principal, owner, now=now)
    decision = dict(schema_version="control_plane_lane_legacy_owner_approval.v1",
                    packet_digest=packet["packet_digest"], principal=principal, owner=owner,
                    approved_action="register_owner_review", approved_at_epoch=now,
                    expires_at_epoch=min(packet["expires_at_epoch"],
                                         now + policy["max_consent_seconds"]),
                    policy_sha256=packet["policy_sha256"], execution_authorized=False,
                    gc_eligible=False, process_fd_references=packet["process_fd_references"],
                    mutations=0)
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
             and approval.get("process_fd_references") == packet.get("process_fd_references")
             and approval.get("approved_action") == "register_owner_review"
             and approval.get("execution_authorized") is False and approval.get("gc_eligible") is False,
             "legacy_owner_approval_invalid")
    _require(type(now) in (int, float) and now >= approval["approved_at_epoch"]
             and now < approval["expires_at_epoch"] and now < packet["expires_at_epoch"],
             "legacy_owner_approval_expired")
    _approved_policy(packet, current_policy_bytes, approval["principal"], approval["owner"], now=now)
    _require(current_generation == packet.get("target_generation"), "legacy_target_changed")
    process_mode, _ = _reference_status(fresh_census, packet.get("selected_path"),
                                        "legacy_owner_references_incomplete")
    _require(process_mode == packet.get("process_fd_references"),
             "legacy_owner_references_incomplete")
    result = dict(schema_version="control_plane_lane_legacy_owner_registration.v1",
                  path=packet["selected_path"], owner=approval["owner"],
                  principal=approval["principal"], packet_digest=packet["packet_digest"],
                  approval_digest=approval["approval_digest"],
                  target_generation=packet["target_generation"],
                  expires_at_epoch=approval["expires_at_epoch"], cleanup="owner_review",
                  classification=("legacy_owner_review" if process_mode == "complete"
                                  else "owner_review_reference_unknown"),
                  process_fd_references=process_mode, gc_eligible=False,
                  references_clear=False, candidate_bytes=None, eta_seconds=None,
                  mutations=0)
    result["registration_digest"] = canonical_digest(result, digest_field="registration_digest")
    return result


class LegacyOwnerStore:
    """Root-owned append-only packet/decision/receipt chain for owner labels."""

    _ID = re.compile(r"[0-9a-f]{32}\Z")
    _ENTRY = re.compile(r"(?:[0-9a-f]{32}\.(?:packet|approval|registration|receipt)\.json|"
                        r"[0-9a-f]{64}\.[0-9a-f]{32}\.head\.json)\Z")
    _TEMP = re.compile(r"\.consent-[0-9a-f]{32}\.tmp\Z")
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
                    if info.st_nlink == 1:
                        owners._protected(info, mode=0o600)
                    else:
                        _require(info.st_nlink == 2 and _root_owned_publication(info),
                                 "legacy_owner_store_unsafe")
                    _require(entry.name == self._LOCK or self._ENTRY.fullmatch(entry.name) is not None
                             or self._TEMP.fullmatch(entry.name) is not None,
                             "legacy_owner_store_unsafe")
                    _require(0 <= info.st_size <= self._CAP, "legacy_owner_store_unsafe")
                    if entry.name == self._LOCK:
                        _require(info.st_size == 0, "legacy_owner_store_unsafe")
                    elif self._ENTRY.fullmatch(entry.name) is not None:
                        names.append(entry.name)
                    size += info.st_size
                    _require(size <= self._MAX_BYTES, "legacy_owner_store_full")
            self.files.verify()
            return sorted(names)
        except (OSError, owners.OwnerCensusConsentError) as error:
            raise LegacyOwnerError("legacy_owner_store_unsafe") from error

    def recover_publication_links(self) -> None:
        """Writer-only repair of this publisher's linked temp, never payload."""
        from . import control_plane_lane_owner_consents as owners

        self._scan()
        try:
            with os.scandir(self.parent) as iterator:
                names = sorted(entry.name for entry in iterator)
            for name in names:
                if self._TEMP.fullmatch(name) is None:
                    continue
                temp = os.stat(name, dir_fd=self.parent, follow_symlinks=False)
                if temp.st_nlink == 1:
                    continue  # Crash before publication; no final was committed.
                _require(temp.st_nlink == 2, "legacy_owner_store_unsafe")
                mates = []
                for candidate in names:
                    self.files.budget.charge("entries")
                    if self._ENTRY.fullmatch(candidate) is None:
                        continue
                    info = os.stat(candidate, dir_fd=self.parent, follow_symlinks=False)
                    if (info.st_dev, info.st_ino) == (temp.st_dev, temp.st_ino):
                        mates.append(candidate)
                _require(len(mates) == 1, "legacy_owner_store_unsafe")
                final_name = mates[0]
                before = os.stat(final_name, dir_fd=self.parent, follow_symlinks=False)
                _require(owners._metadata(before) == owners._metadata(temp)
                         and _root_owned_publication(temp)
                         and 0 < temp.st_size <= self._CAP,
                         "legacy_owner_store_unsafe")
                self.files.verify()
                _require(owners._metadata(os.stat(name, dir_fd=self.parent, follow_symlinks=False))
                         == owners._metadata(temp)
                         and owners._metadata(os.stat(final_name, dir_fd=self.parent, follow_symlinks=False))
                         == owners._metadata(before), "legacy_owner_store_unsafe")
                os.unlink(name, dir_fd=self.parent)
                os.fsync(self.parent)
                final = os.stat(final_name, dir_fd=self.parent, follow_symlinks=False)
                owners._protected(final, mode=0o600)
                _require((final.st_dev, final.st_ino, final.st_size)
                         == (before.st_dev, before.st_ino, before.st_size), "legacy_owner_store_unsafe")
            self._scan()
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

        _require(self._ENTRY.fullmatch(name) is not None, "legacy_owner_record_invalid")
        descriptor = None
        try:
            # The protected parent chain is already pinned in this session.
            # Reopening the whole chain for every head exhausts the 16-root
            # descriptor budget before even a handful of committed labels.
            self.files.verify()
            descriptor = self.files.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                         parent=self.parent)
            before = os.fstat(descriptor)
            named = os.stat(name, dir_fd=self.parent, follow_symlinks=False)
            _require(owners._metadata(before) == owners._metadata(named)
                     and 0 < before.st_size <= self._CAP
                     and before.st_nlink == 1, "legacy_owner_record_invalid")
            owners._protected(before, mode=0o600)
            raw = self.files.read_bytes(descriptor, self._CAP)
            _require(len(raw) == before.st_size
                     and owners._metadata(os.fstat(descriptor)) == owners._metadata(before)
                     and owners._metadata(os.stat(name, dir_fd=self.parent, follow_symlinks=False))
                     == owners._metadata(before), "legacy_owner_record_invalid")
            self.files.budget.charge("entries")
            value = retained._document(raw, self._CAP, _work_budget=self.files.budget)
            _require(type(value) is dict and raw == self._payload(value),
                     "legacy_owner_record_invalid")
            self.files.verify()
            return value
        except (OSError, owners.OwnerCensusConsentError, retained.CensusDecisionError) as error:
            raise LegacyOwnerError("legacy_owner_record_invalid") from error
        finally:
            if descriptor is not None:
                self.files.close(descriptor)

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


def _registry_root(config) -> Path:
    """Derived from the protected installed store; never caller selected."""
    return Path(config.owner_consent_store).parent / "legacy-owner-registrations"


@contextmanager
def _installed_session(installed_config_path: str, monotonic, *, write: bool = False,
                       max_seconds: float = 5.0):
    from . import control_plane_lane_owner_consents as owners
    from .control_plane_reference_budget import ReferenceCollectionBudget, ReferenceCollectionBudgetError

    _require(os.geteuid() == 0, "legacy_owner_root_required")
    budget = ReferenceCollectionBudget(monotonic=monotonic,
                                       time_budget_seconds=max_seconds, values_limit=10_000)
    files = owners._Files(budget, raw_cap=2 * 1024 * 1024)
    try:
        config = owners._installed_config(files, installed_config_path)
        store = LegacyOwnerStore(files, _registry_root(config))
        if write:
            store.recover_publication_links()
        yield files, budget, config, store
        files.verify()
    except (owners.OwnerCensusConsentError, ReferenceCollectionBudgetError) as error:
        raise LegacyOwnerError("legacy_owner_installed_authority_unavailable") from error
    finally:
        try:
            files.finish()
        finally:
            budget.close()


def _policy_bytes(files, config) -> bytes:
    from . import control_plane_lane_owner_consents as owners

    raw, _ = files.read(config.lane_owner_policy_file, cap=owners.MAX_POLICY_BYTES,
                        protected=True, mode=0o600)
    return raw


def _load_old_consent(files, budget, config, *, consent_id: str,
                      expected_sha256: str, expected_size_bytes: int, now: float) -> dict:
    from . import control_plane_lane_owner_consents as owners

    _require(isinstance(consent_id, str) and re.fullmatch(r"[0-9a-f]{32}", consent_id) is not None
             and isinstance(expected_sha256, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", expected_sha256)
             and type(expected_size_bytes) is int and 0 < expected_size_bytes <= owners.MAX_RECORD_BYTES,
             "legacy_owner_consent_selector_invalid")
    raw, _ = files.read(Path(config.owner_consent_store) / (consent_id + ".json"),
                        cap=owners.MAX_RECORD_BYTES, protected=True, mode=0o600)
    _require(len(raw) == expected_size_bytes
             and "sha256:" + hashlib.sha256(raw).hexdigest() == expected_sha256,
             "legacy_owner_consent_changed")
    record = owners._record(raw, consent_id, _policy_bytes(files, config),
                            owners._roots(config, budget), now, budget)
    return record | dict(_raw_record_sha256=expected_sha256,
                         _raw_record_size_bytes=expected_size_bytes)


def _recheck_packet_consent(files, budget, config, packet: dict, *, now: float) -> None:
    old = packet.get("old_consent")
    _require(isinstance(old, dict), "legacy_owner_consent_changed")
    current = _load_old_consent(files, budget, config,
                                consent_id=old.get("consent_id"),
                                expected_sha256=old.get("record_sha256"),
                                expected_size_bytes=old.get("record_size_bytes"), now=now)
    _require(current.get("consent_digest") == old.get("sha256")
             and current.get("census") == old.get("census")
             and current.get("annotations") == old.get("annotations")
             and current.get("principal") == packet.get("principal")
             and current.get("policy_sha256") == packet.get("policy_sha256"),
             "legacy_owner_consent_changed")


_GC_UNIT = Path("/etc/systemd/system/blueprint-control-plane-storage-gc.service")
_REFERENCE_KEYS = frozenset({"BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS",
                             "BLUEPRINT_CONTROL_PLANE_GC_EVIDENCE_ROOTS",
                             "BLUEPRINT_CONTROL_PLANE_GC_SETTLEMENT_ROOTS",
                             "BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT"})


def _reference_settings(files, config) -> dict[str, tuple[Path, ...] | Path]:
    """Read current root-controlled GC selections; empty caller claims are ignored."""
    unit_raw, _ = files.read(_GC_UNIT, cap=65536, protected=True)
    env_raw, _ = files.read(config.experiment_gc_environment_file, cap=65536, protected=True)
    selected = {}
    try:
        for source in (unit_raw, env_raw):
            for line in source.decode("utf-8").splitlines():
                value = line.strip()
                if value.startswith("Environment="):
                    value = value[len("Environment="):]
                key, separator, raw = value.partition("=")
                if separator and key in _REFERENCE_KEYS:
                    raw = raw.strip().strip('"').strip("'")
                    _require(raw and not any(char in raw for char in ("$", "`", "\\", "\n", "\r", " ", "\t")),
                             "legacy_owner_references_incomplete")
                    selected[key] = raw
        def paths(key):
            values = selected[key].split(":")
            _require(0 < len(values) <= 64 and all(values), "legacy_owner_references_incomplete")
            result = tuple(Path(value) for value in values)
            for value in result:
                _absolute(value)
            return result
        queues = paths("BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS")
        active = paths("BLUEPRINT_CONTROL_PLANE_GC_EVIDENCE_ROOTS") + paths(
            "BLUEPRINT_CONTROL_PLANE_GC_SETTLEMENT_ROOTS")
        pins = paths("BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT")
        _require(len(pins) == 1, "legacy_owner_references_incomplete")
        return dict(queue_roots=queues, active_run_roots=active, pins_root=pins[0])
    except (KeyError, UnicodeError, LegacyOwnerError):
        raise LegacyOwnerError("legacy_owner_references_incomplete") from None


def _config_identity(config) -> dict:
    return dict(vars(config))


def _remaining(monotonic, deadline: float, limit: float) -> float:
    value = min(limit, deadline - monotonic())
    _require(value > 0, "legacy_owner_budget_exhausted")
    return value


def _effective_now(now: float, started: float, monotonic, wall_clock=None) -> float:
    elapsed = monotonic() - started
    _require(0 <= elapsed <= MAX_SECONDS and math.isfinite(now + elapsed),
             "legacy_owner_budget_exhausted")
    current = now + elapsed
    if wall_clock is not None:
        observed = wall_clock()
        _require(type(observed) in (int, float) and math.isfinite(observed),
                 "legacy_owner_budget_exhausted")
        current = max(current, observed)
    return current


def _same_authority(files, config, *, expected_config: dict,
                    expected_settings: dict | None = None,
                    expected_policy: bytes | None = None) -> None:
    _require(_config_identity(config) == expected_config,
             "legacy_owner_installed_authority_changed")
    if expected_settings is not None:
        _require(_reference_settings(files, config) == expected_settings,
                 "legacy_owner_references_incomplete")
    if expected_policy is not None:
        _require(_policy_bytes(files, config) == expected_policy,
                 "legacy_owner_policy_changed")


def _fresh_census(config, selected: dict, *, now: float, max_seconds: float = 120) -> dict:
    from .control_plane_lane_scratch_census import build_census

    report = build_census(work_root=Path(config.lane_scratch_work_root).parent,
                          inputs_root=Path(config.lane_scratch_inputs_root).parent,
                          process_root=Path("/proc"), pins_root=selected["pins_root"],
                          queue_roots=selected["queue_roots"],
                          active_run_roots=selected["active_run_roots"],
                          release_link=Path(config.active_release_link), now=now, max_seconds=max_seconds)
    _require((report.get("status") == "complete" and report.get("scan_errors") == [])
             or (report.get("status") == "incomplete"
                 and report.get("scan_errors") == ["process_inventory_unreadable"]
                 and type(report.get("candidate_count")) is int
                 and isinstance(report.get("rows"), list)
                 and report["candidate_count"] == len(report["rows"])),
             "legacy_owner_references_incomplete")
    return report


def _snapshot_for(config, path: str, *, max_seconds: float = MAX_SECONDS) -> dict:
    roots = (Path(config.lane_scratch_work_root).parent,
             Path(config.lane_scratch_inputs_root).parent)
    return snapshot_generation(path, allowed_roots=roots, max_seconds=max_seconds)


def issue_version_packet(*, consent_id: str, consent_sha256: str, consent_size_bytes: int,
                         selected_path: str, installed_config_path: str, now: float,
                         monotonic=time.monotonic, wall_clock=None) -> dict:
    """Persist a reviewable target packet; no owner decision is inferred."""
    started = monotonic()
    deadline = started + MAX_SECONDS
    with _installed_session(installed_config_path, monotonic, write=True,
                            max_seconds=_remaining(monotonic, deadline, 5)) as (files, budget, config, store):
        consent = _load_old_consent(files, budget, config, consent_id=consent_id,
                                    expected_sha256=consent_sha256,
                                    expected_size_bytes=consent_size_bytes, now=now)
        settings = _reference_settings(files, config)
        policy, identity = _policy_bytes(files, config), _config_identity(config)
    census = _fresh_census(config, settings, now=now,
                           max_seconds=_remaining(monotonic, deadline, 120))
    generation = _snapshot_for(config, selected_path,
                               max_seconds=_remaining(monotonic, deadline, MAX_SECONDS))
    packet = build_version_packet(consent, selected_path=selected_path,
                                  generation=generation, fresh_census=census, now=now)
    _require(_effective_now(now, started, monotonic, wall_clock) < packet["expires_at_epoch"],
             "legacy_owner_packet_expired")
    packet_id = secrets.token_hex(16)
    with _installed_session(installed_config_path, monotonic, write=True,
                            max_seconds=_remaining(monotonic, deadline, 5)) as (files, budget, current, store):
        _same_authority(files, current, expected_config=identity,
                        expected_settings=settings, expected_policy=policy)
        _require(_load_old_consent(files, budget, current, consent_id=consent_id,
                                   expected_sha256=consent_sha256,
                                   expected_size_bytes=consent_size_bytes,
                                   now=_effective_now(now, started, monotonic, wall_clock)) == consent,
                 "legacy_owner_consent_changed")
        _require(_effective_now(now, started, monotonic, wall_clock) < packet["expires_at_epoch"],
                 "legacy_owner_packet_expired")
        store.publish(packet_id, "packet", packet)
    _require(_snapshot_for(config, selected_path,
                           max_seconds=_remaining(monotonic, deadline, MAX_SECONDS)) == generation,
             "legacy_target_changed")
    return dict(status="packet_ready_for_separate_owner_review", packet_id=packet_id,
                packet=packet, packet_digest=packet["packet_digest"],
                execution_authorized=False, gc_eligible=False, target_mutations=0)


def issue_generation_approval(*, packet_id: str, ack_packet_digest: str,
                              principal: str, owner: str, installed_config_path: str,
                              now: float, monotonic=time.monotonic,
                              ack_process_fd_unknown: bool = False, wall_clock=None) -> dict:
    """Distinct root/owner action requiring the exact packet digest as input."""
    started = monotonic()
    deadline = started + MAX_SECONDS
    with _installed_session(installed_config_path, monotonic, write=True,
                            max_seconds=_remaining(monotonic, deadline, 5)) as (files, budget, config, store):
        packet = store.read(packet_id, "packet")
        _recheck_packet_consent(files, budget, config, packet,
                                now=_effective_now(now, started, monotonic, wall_clock))
        policy, identity = _policy_bytes(files, config), _config_identity(config)
        approval = approve_version_packet(packet, ack_packet_digest=ack_packet_digest,
                                          current_policy_bytes=policy,
                                          principal=principal, owner=owner,
                                          now=_effective_now(now, started, monotonic, wall_clock),
                                          ack_process_fd_unknown=ack_process_fd_unknown)
    _require(_snapshot_for(config, packet["selected_path"],
                           max_seconds=_remaining(monotonic, deadline, MAX_SECONDS)) == packet["target_generation"],
             "legacy_target_changed")
    with _installed_session(installed_config_path, monotonic, write=True,
                            max_seconds=_remaining(monotonic, deadline, 5)) as (files, budget, current, store):
        _same_authority(files, current, expected_config=identity, expected_policy=policy)
        _require(store.read(packet_id, "packet") == packet, "legacy_owner_record_changed")
        _recheck_packet_consent(files, budget, current, packet,
                                now=_effective_now(now, started, monotonic, wall_clock))
        _require(_effective_now(now, started, monotonic, wall_clock) < approval["expires_at_epoch"],
                 "legacy_owner_approval_expired")
        store.publish(packet_id, "approval", approval)
    _require(_snapshot_for(config, packet["selected_path"],
                           max_seconds=_remaining(monotonic, deadline, MAX_SECONDS)) == packet["target_generation"],
             "legacy_target_changed")
    return dict(status="owner_generation_approval_recorded", packet_id=packet_id,
                packet_digest=packet["packet_digest"],
                approval_digest=approval["approval_digest"],
                expires_at_epoch=approval["expires_at_epoch"],
                execution_authorized=False, gc_eligible=False, target_mutations=0)


def apply_owner_review(*, packet_id: str, installed_config_path: str,
                       now: float, monotonic=time.monotonic, wall_clock=None) -> dict:
    """Publish only a protected external owner label and recoverable receipt."""
    started = monotonic()
    deadline = started + MAX_SECONDS
    with _installed_session(installed_config_path, monotonic, write=True,
                            max_seconds=_remaining(monotonic, deadline, 5)) as (files, budget, config, store):
        packet, approval = store.read(packet_id, "packet"), store.read(packet_id, "approval")
        _recheck_packet_consent(files, budget, config, packet,
                                now=_effective_now(now, started, monotonic, wall_clock))
        settings, policy = _reference_settings(files, config), _policy_bytes(files, config)
        identity = _config_identity(config)
    path = packet["selected_path"]
    current = _snapshot_for(config, path,
                            max_seconds=_remaining(monotonic, deadline, MAX_SECONDS))
    census = _fresh_census(config, settings, now=now,
                           max_seconds=_remaining(monotonic, deadline, 120))
    registration = validate_registration(packet, approval, current_generation=current,
                                         fresh_census=census, current_policy_bytes=policy,
                                         now=_effective_now(now, started, monotonic, wall_clock))
    receipt = dict(schema_version="control_plane_lane_legacy_owner_receipt.v1",
                   packet_digest=packet["packet_digest"],
                   approval_digest=approval["approval_digest"],
                   registration=registration, target_mutations=0,
                   candidate_bytes=None, eta_seconds=None)
    for stage in ("registration", "receipt", "head"):
        _require(_snapshot_for(config, path,
                               max_seconds=_remaining(monotonic, deadline, MAX_SECONDS)) == current,
                 "legacy_target_changed")
        if stage == "head":
            latest = _fresh_census(config, settings, now=now,
                                   max_seconds=_remaining(monotonic, deadline, 120))
            _require(validate_registration(packet, approval, current_generation=current,
                                           fresh_census=latest, current_policy_bytes=policy,
                                           now=_effective_now(now, started, monotonic, wall_clock)) == registration,
                     "legacy_owner_references_incomplete")
        with _installed_session(installed_config_path, monotonic, write=True,
                                max_seconds=_remaining(monotonic, deadline, 5)) as (files, budget, active, store):
            _same_authority(files, active, expected_config=identity,
                            expected_settings=settings, expected_policy=policy)
            _require(store.read(packet_id, "packet") == packet
                     and store.read(packet_id, "approval") == approval,
                     "legacy_owner_record_changed")
            _recheck_packet_consent(files, budget, active, packet,
                                    now=_effective_now(now, started, monotonic, wall_clock))
            for head in store.committed_heads():
                _require(head["path"] != path or head["packet_id"] == packet_id,
                         "legacy_owner_active_conflict")
            if stage == "registration":
                store.publish(packet_id, stage, registration)
            elif stage == "receipt":
                _require(store.read(packet_id, "registration") == registration,
                         "legacy_owner_record_changed")
                store.publish(packet_id, stage, receipt)
            else:
                _require(store.read(packet_id, "registration") == registration
                         and store.read(packet_id, "receipt") == receipt,
                         "legacy_owner_record_changed")
                _require(validate_registration(packet, approval, current_generation=current,
                                               fresh_census=latest, current_policy_bytes=policy,
                                               now=_effective_now(now, started, monotonic, wall_clock))
                         == registration, "legacy_owner_approval_expired")
                store.publish_head(path, packet_id, registration)
        _require(_snapshot_for(config, path,
                               max_seconds=_remaining(monotonic, deadline, MAX_SECONDS)) == current,
                 "legacy_target_changed")
    return dict(status="legacy_owner_review_registered", packet_id=packet_id,
                path=path, owner=registration["owner"],
                expires_at_epoch=registration["expires_at_epoch"],
                gc_eligible=False, references_clear=False, target_mutations=0,
                candidate_bytes=None, eta_seconds=None)


def observe_owner_review(*, installed_config_path: str, now: float,
                         monotonic=time.monotonic, max_seconds: float | None = None,
                         wall_clock=None) -> dict:
    """Fresh owner census with authenticated legacy labels, never GC authority."""
    _require(max_seconds is None or (type(max_seconds) in (int, float)
             and 0 < max_seconds <= MAX_SECONDS), "legacy_owner_options_invalid")
    started = monotonic()
    deadline = started + (MAX_SECONDS if max_seconds is None else max_seconds)

    def remaining(limit: float) -> float:
        return _remaining(monotonic, deadline, limit)

    def incomplete(code: str) -> dict:
        return dict(schema_version="control_plane_lane_legacy_owner_survey.v1",
                    status="incomplete", rows=[], scan_errors=[code],
                    observed_owner_count=0, gc_eligible=False,
                    references_clear=False, candidate_bytes=None,
                    eta_seconds=None, mutations=0)

    try:
        with _installed_session(installed_config_path, monotonic,
                                max_seconds=remaining(5)) as (files, _, config, store):
            settings, current_policy = _reference_settings(files, config), _policy_bytes(files, config)
            identity = _config_identity(config)
        census = _fresh_census(config, settings, now=now, max_seconds=remaining(120))
        with _installed_session(installed_config_path, monotonic,
                                max_seconds=remaining(5)) as (files, _, active, store):
            _same_authority(files, active, expected_config=identity,
                            expected_settings=settings, expected_policy=current_policy)
            heads = store.committed_heads()
    except LegacyOwnerError as error:
        return incomplete(str(error))
    rows = [dict(row) for row in census["rows"]]
    by_path = {row["path"]: row for row in rows}
    original_by_path = {row["path"]: row for row in census["rows"]}
    possible: dict[str, list[tuple[dict, float]]] = {}
    blockers = []
    for head in heads:
        try:
            remaining(MAX_SECONDS)
        except LegacyOwnerError as error:
            return incomplete(str(error))
        path, packet_id = head["path"], head["packet_id"]
        if path not in by_path:
            continue
        try:
            with _installed_session(installed_config_path, monotonic,
                                    max_seconds=remaining(5)) as (files, budget, active, store):
                _same_authority(files, active, expected_config=identity,
                                expected_settings=settings, expected_policy=current_policy)
                packet = store.read(packet_id, "packet")
                _recheck_packet_consent(files, budget, active, packet,
                                        now=_effective_now(now, started, monotonic, wall_clock))
                approval = store.read(packet_id, "approval")
                recorded = store.read(packet_id, "registration")
                _require(store._read(store._head_name(path, packet_id)) == head,
                         "legacy_owner_record_changed")
            current = _snapshot_for(config, path, max_seconds=remaining(MAX_SECONDS))
            checked = validate_registration(packet, approval, current_generation=current,
                                            fresh_census=census,
                                            current_policy_bytes=current_policy,
                                            now=_effective_now(now, started, monotonic, wall_clock))
            _require(recorded == checked, "legacy_owner_record_invalid")
            with _installed_session(installed_config_path, monotonic,
                                    max_seconds=remaining(5)) as (files, budget, active, store):
                _same_authority(files, active, expected_config=identity,
                                expected_settings=settings, expected_policy=current_policy)
                _require(store.read(packet_id, "packet") == packet
                         and store.read(packet_id, "approval") == approval
                         and store.read(packet_id, "registration") == recorded
                         and store._read(store._head_name(path, packet_id)) == head,
                         "legacy_owner_record_changed")
                _recheck_packet_consent(files, budget, active, packet,
                                        now=_effective_now(now, started, monotonic, wall_clock))
            _require(_snapshot_for(config, path, max_seconds=remaining(MAX_SECONDS)) == current,
                     "legacy_target_changed")
            _require(_effective_now(now, started, monotonic, wall_clock)
                     < approval["expires_at_epoch"]
                     and _effective_now(now, started, monotonic, wall_clock)
                     < packet["expires_at_epoch"], "legacy_owner_approval_expired")
            possible.setdefault(path, []).append((
                checked, min(packet["expires_at_epoch"], approval["expires_at_epoch"])))
        except LegacyOwnerError as error:
            blockers.append(str(error))
    try:
        remaining(MAX_SECONDS)
    except LegacyOwnerError as error:
        return incomplete(str(error))
    applied = 0
    projected_expiries: dict[str, float] = {}
    for path, candidates in possible.items():
        if len(candidates) != 1:
            blockers.append("legacy_owner_active_conflict")
            continue
        record, expires_at = candidates[0]
        try:
            current_now = _effective_now(now, started, monotonic, wall_clock)
        except LegacyOwnerError as error:
            return incomplete(str(error))
        if current_now >= expires_at:
            blockers.append("legacy_owner_approval_expired")
            continue
        row = by_path[path]
        row.update(owner=record["owner"], owner_decision="owner_review",
                   approved_expiry=record["expires_at_epoch"],
                   classification=record["classification"],
                   process_fd_references=record["process_fd_references"],
                   owner_source="protected_second_generation_approval",
                   gc_eligible=False, references_clear=False,
                   candidate_bytes=None, eta_seconds=None)
        projected_expiries[path] = expires_at
        applied += 1
    try:
        final_now = _effective_now(now, started, monotonic, wall_clock)
    except LegacyOwnerError as error:
        return incomplete(str(error))
    for path, expires_at in projected_expiries.items():
        if final_now >= expires_at:
            by_path[path].clear()
            by_path[path].update(original_by_path[path])
            blockers.append("legacy_owner_approval_expired")
            applied -= 1
    report = dict(census)
    report.update(schema_version="control_plane_lane_legacy_owner_survey.v1",
                  rows=rows, observed_owner_count=applied,
                  legacy_owner_blockers=sorted(set(blockers)),
                  gc_eligible=False, references_clear=False,
                  candidate_bytes=None, eta_seconds=None, mutations=0)
    return report


def main(argv=None) -> int:
    """Root operator entrypoint; every mode keeps legacy payload bytes in place."""
    import argparse

    class Parser(argparse.ArgumentParser):
        def error(self, message):
            raise LegacyOwnerError("legacy_owner_options_invalid")

    parser = Parser(allow_abbrev=False)
    parser.add_argument("mode", choices=("packet", "approve", "apply", "report"))
    parser.add_argument("--consent-id")
    parser.add_argument("--consent-sha256")
    parser.add_argument("--consent-size-bytes", type=int)
    parser.add_argument("--selected-path")
    parser.add_argument("--packet-id")
    parser.add_argument("--ack-packet-digest")
    parser.add_argument("--ack-process-fd-unknown", action="store_true")
    parser.add_argument("--principal")
    parser.add_argument("--owner")
    parser.add_argument("--door-config", default="/etc/blueprint-operator-door/door.json")
    try:
        args = parser.parse_args(argv)
        options = {key for key in ("consent_id", "consent_sha256", "consent_size_bytes",
                                   "selected_path", "packet_id", "ack_packet_digest",
                                   "principal", "owner") if getattr(args, key) is not None}
        expected = {"packet": {"consent_id", "consent_sha256", "consent_size_bytes", "selected_path"},
                    "approve": {"packet_id", "ack_packet_digest", "principal", "owner"},
                    "apply": {"packet_id"}, "report": set()}
        _require(options == expected[args.mode], "legacy_owner_options_invalid")
        _require(args.mode == "approve" or not args.ack_process_fd_unknown,
                 "legacy_owner_options_invalid")
        selected = dict(installed_config_path=args.door_config, now=time.time(),
                        monotonic=time.monotonic, wall_clock=time.time)
        if args.mode == "packet":
            result = issue_version_packet(consent_id=args.consent_id,
                                          consent_sha256=args.consent_sha256,
                                          consent_size_bytes=args.consent_size_bytes,
                                          selected_path=args.selected_path, **selected)
        elif args.mode == "approve":
            result = issue_generation_approval(packet_id=args.packet_id,
                                               ack_packet_digest=args.ack_packet_digest,
                                               principal=args.principal, owner=args.owner,
                                               ack_process_fd_unknown=args.ack_process_fd_unknown,
                                               **selected)
        elif args.mode == "apply":
            result = apply_owner_review(packet_id=args.packet_id, **selected)
        else:
            result = observe_owner_review(**selected)
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except LegacyOwnerError as error:
        print(json.dumps(dict(status="refused", blockers=[str(error)],
                              gc_eligible=False, references_clear=False,
                              candidate_bytes=None, eta_seconds=None,
                              target_mutations=0), sort_keys=True, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
