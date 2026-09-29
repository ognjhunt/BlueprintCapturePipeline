"""Current authenticated experiment target lifetime for finite enrolled consumers.

Admission decodes under one native bounded budget. Retained proof checks allocate
no new records; an authority change refuses this admission instead of adopting it.
"""
from __future__ import annotations

import contextvars
import fcntl
import grp
import os
import re
import stat
import time
from contextlib import contextmanager
from pathlib import Path

from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch as scratch
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_experiment_authority import _current, _read
from .control_plane_lane_experiment_publication import _BirthFiles
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _epoch, _require, _valid_digest
from .control_plane_reference_budget import ReferenceCollectionBudget, ReferenceCollectionBudgetError
from .control_plane_scratch_lifetime import LANE_ROOTS, LeasedScratchUse
from .decision_evidence_contracts import canonical_digest

_CURRENT_USE = contextvars.ContextVar("registered_experiment_current_use", default=None)

PRODUCER_CONFIG_PATH = Path("/etc/blueprint-operator-door/door.json")

AUTHORITY_ROOT = Path("/var/lib/blueprint-operator-door/experiment-authority")


def _blueprint_gid():
    try:
        return grp.getgrnam("blueprint").gr_gid
    except KeyError:
        raise OwnerTargetVersionError("experiment_account_missing") from None


def _canonical_reader_path(value, *, _budget=None):
    """Reject unsafe spellings before payload access, without resolving a link.

    Named no-follow metadata is finite pre-call evidence, never permission or
    an atomic fence against another thread changing pathname components.
    """
    budget = _budget if _budget is not None else ReferenceCollectionBudget(values_limit=10000)
    try:
        budget.tick()
        _require(isinstance(value, Path) and len(value.parts) <= 64 and '..' not in value.parts,
                 'experiment_consumer_path_unsafe')
        text = str(value)
        _require(len(text) <= 4096 and len(text.encode('utf-8')) <= 4096,
                 'experiment_consumer_path_unsafe')
        budget.charge('raw_bytes', len(text.encode('utf-8')))
        absolute = value if value.is_absolute() else Path.cwd() / value
        _require(len(absolute.parts) <= 64, 'experiment_consumer_path_unsafe')
        parent = Path(absolute.anchor)
        for component in absolute.parts[1:]:
            budget.charge('values')
            parent = parent / component
            try:
                info = os.stat(parent, follow_symlinks=False)
            except FileNotFoundError:
                break
            _require(not stat.S_ISLNK(info.st_mode), 'experiment_consumer_path_unsafe')
        budget.tick()
        return absolute
    except (OSError, ReferenceCollectionBudgetError):
        raise OwnerTargetVersionError('experiment_consumer_path_unsafe') from None
    finally:
        if _budget is None:
            budget.close()


def registered_target(output, roots, *, _budget=None):
    """Marker-independent reservation; alias spellings cannot enter legacy."""
    if not isinstance(output, Path):
        return None
    selected = _canonical_reader_path(output, _budget=_budget)
    for root in roots:
        if selected.is_relative_to(root):
            parts = selected.relative_to(root).parts
            if len(parts) >= 2 and parts[0] in ('g1', 'arena') and parts[1].startswith('registered-'):
                _require(re.fullmatch(r'registered-[0-9a-f]{32}', parts[1]) is not None,
                         'experiment_consumer_path_unsafe')
                return root / parts[0] / parts[1]
    return None


def require_registered_use(output, use, roots):
    target = registered_target(output, roots)
    if target is not None:
        _require(type(use) is RegisteredExperimentUse and use.path == target and not use._closed,
                 "experiment_consumer_authority_required")
        use.check()
    elif use is not None:
        raise OwnerTargetVersionError("experiment_consumer_authority_required")
    return target


def _restoration(files, public, entry, gid, issued):
    """Readable root-owned projection; never requires private restore records."""
    selected = entry["restoration"]
    if selected is None:
        return None
    name = "restoration-" + selected["sha256"][7:] + ".json"
    raw, record = _read(files, public, name, 8192, gid)
    _require(len(raw) == selected["size_bytes"]
             and retained._digest(raw, _work_budget=files.budget) == selected["sha256"],
             "experiment_restoration_changed")
    certificate = retained._document(raw, 8192, _work_budget=files.budget)
    fields = {"schema_version", "restoration_id", "intent_id", "generation", "birth", "target_identity",
              "new_lease", "manifest", "restored_at_epoch", "lease_expires_at_epoch", "certificate_digest"}
    _require(set(certificate) == fields
             and certificate["schema_version"] == "control_plane_lane_experiment_restoration_certificate.v1"
             and certificate["certificate_digest"] == canonical_digest(certificate, digest_field="certificate_digest")
             and all(certificate[key] == entry[key] for key in ("intent_id", "generation", "birth", "target_identity"))
             and certificate["new_lease"] == entry["lease"]
             and certificate["restoration_id"] == entry["operation_id"]
             and _epoch(certificate["restored_at_epoch"])
             and certificate["restored_at_epoch"] <= issued < certificate["lease_expires_at_epoch"] == entry["expires_at_epoch"],
             "experiment_restoration_changed")
    manifest = certificate["manifest"]
    _require(type(manifest) is dict and set(manifest) == {"sha256", "size_bytes"}
             and _valid_digest(manifest["sha256"]) and type(manifest["size_bytes"]) is int
             and 0 < manifest["size_bytes"] <= 1048576, "experiment_restoration_changed")
    return record


class RegisteredExperimentUse(LeasedScratchUse):
    """Borrowed scopes retain the same target SH; they never close the parent."""
    @classmethod
    def admit(cls, target, *, expected_birth=None, expected_generation=None, now=time.time,
              _producer_request_paths=None, _producer_config_path=None, _producer_bootstrap_path=None, _arena_tag=None,
              _metadata_budget=None):
        _require(_metadata_budget is None or type(_metadata_budget) is ReferenceCollectionBudget
                 and _metadata_budget.duration == 5.0 and _metadata_budget.limits['values'] == 10000
                 and not _metadata_budget.closed and _metadata_budget.failure is None,
                 'experiment_consumer_path_unsafe')
        files = _BirthFiles(_metadata_budget if _metadata_budget is not None else
                            ReferenceCollectionBudget(values_limit=10000))
        result = None
        accepted = False
        try:
            _require((isinstance(target, Path) and registered_target(target, LANE_ROOTS, _budget=files.budget) == target
                      and _arena_tag is None)
                     or target is None and isinstance(_arena_tag, str) and re.fullmatch(r"r[1-9][0-9]{0,5}", _arena_tag)
                     and _producer_request_paths is None and _producer_config_path is None and _producer_bootstrap_path is None,
                     "experiment_consumer_authority_required")
            gid = _blueprint_gid()
            public, _ = files.parent(AUTHORITY_ROOT / "HEAD.json")
            info = os.fstat(public)
            _require(info.st_uid == 0 and info.st_gid == gid and stat.S_IMODE(info.st_mode) == 0o750,
                     "experiment_authority_unsafe")
            lock = files.open(".authority.lock", os.O_RDONLY | os.O_NONBLOCK, parent=public)
            lock_info = files.acquired[lock]
            _require(lock_info.st_uid == 0 and lock_info.st_gid == gid and stat.S_IMODE(lock_info.st_mode) == 0o640
                     and lock_info.st_nlink == 1 and lock_info.st_size == 0, "experiment_authority_unsafe")
            files.location(public)
            files.proof(lock)
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            current = _current(files, public, gid)
            _require(current is not None, "experiment_authority_missing")
            head, authority, head_record = current
            issued = now()
            _require(_epoch(issued) and authority["state"] == "enabled"
                     and authority["issued_at_epoch"] <= issued < authority["expires_at_epoch"],
                     "experiment_consumer_inactive")
            arena_selection_record = None
            if _arena_tag is not None:
                raw, arena_selection_record = _read(files, public, "arena-selection-" + _arena_tag + ".json", 4096, gid)
                selected = retained._document(raw, 4096, _work_budget=files.budget)
                _require(set(selected) == {"schema_version", "tag", "intent_id", "generation", "birth", "lease",
                                          "target_identity", "reference_kind", "reference_value", "selection_digest"}
                         and selected["schema_version"] == "control_plane_lane_arena_selection.v1"
                         and selected["selection_digest"] == canonical_digest(selected, digest_field="selection_digest")
                         and selected["tag"] == _arena_tag
                         and owners._matches(selected["intent_id"], owners._CONSENT_ID)
                         and owners._matches(selected["generation"], owners._CONSENT_ID)
                         and selected["reference_kind"] == "run_ref"
                         and selected["reference_value"] == "arena-launch-" + _arena_tag,
                         "experiment_arena_selection_invalid")
                expected_birth, expected_generation = selected["birth"], selected["generation"]
                target = LANE_ROOTS[1] / "arena" / ("registered-" + selected["intent_id"])
            root = next(root for root in LANE_ROOTS if target.is_relative_to(root))
            root_selector = "work" if root == LANE_ROOTS[0] else "inputs"
            rows = [entry for entry in authority["enrollments"] if entry["root"] == root_selector
                    and entry["name"] == target.name and entry["lane"] == target.parent.name]
            _require(len(rows) == 1, "experiment_consumer_authority_required")
            entry = rows[0]
            _require(entry["state"] == "active" and issued < entry["expires_at_epoch"]
                     and (expected_birth is None or expected_birth == entry["birth"])
                     and (expected_generation is None or expected_generation == entry["generation"]),
                     "experiment_consumer_inactive")
            birth_raw, birth_record = _read(files, public, entry["intent_id"] + ".birth.json", 32768, gid)
            _require(len(birth_raw) == entry["birth"]["size_bytes"]
                     and retained._digest(birth_raw, _work_budget=files.budget) == entry["birth"]["sha256"],
                     "experiment_birth_changed")
            birth = retained._document(birth_raw, 32768, _work_budget=files.budget)
            _require(birth["schema_version"] == "control_plane_lane_experiment_birth.v1"
                     and birth["birth_digest"] == canonical_digest(birth, digest_field="birth_digest")
                     and all(birth[key] == entry[key] for key in ("intent_id", "generation", "root", "lane", "name",
                                                                "target_identity", "owner")), "experiment_birth_changed")
            if entry["lane"] == "arena":
                _require(birth["participant_profile"] == "arena_owner_review.v1" and entry["root"] == "inputs"
                         and birth["writer_scope"] == "arena_construction_launch_chain.v1"
                         and birth["class_intent"] == "evidence" and birth["cleanup"] == "owner_review"
                         and birth["reference_kind"] == "run_ref"
                         and isinstance(birth["reference_value"], str)
                         and re.fullmatch(r"arena-launch-r[1-9][0-9]{0,5}", birth["reference_value"])
                         and (_arena_tag is None or birth["reference_value"] == selected["reference_value"]
                              and entry["lease"] == selected["lease"] and entry["target_identity"] == selected["target_identity"]),
                         "experiment_arena_selection_invalid")
            restoration_record = _restoration(files, public, entry, gid, issued)
            files.proof(lock)
            fcntl.flock(lock, fcntl.LOCK_UN)
            parent, _ = files.parent(target / scratch.LEASE_FILE)
            files.location(parent)
            original = os.fstat(parent)
            _require((original.st_dev, original.st_ino) == (entry["target_identity"]["dev"], entry["target_identity"]["ino"]),
                     "experiment_target_changed")
            files.proof(parent)
            fcntl.flock(parent, fcntl.LOCK_SH | fcntl.LOCK_NB)
            lease_raw, lease_record = files.read(target / scratch.LEASE_FILE, cap=scratch.MAX_LEASE_BYTES)
            _require(len(lease_raw) == entry["lease"]["size_bytes"]
                     and retained._digest(lease_raw, _work_budget=files.budget) == entry["lease"]["sha256"],
                     "experiment_lease_changed")
            lease = retained._document(lease_raw, scratch.MAX_LEASE_BYTES, _work_budget=files.budget)
            _require(lease["schema_version"] == scratch.SCHEMA_VERSION and scratch._lease_fields_valid(lease)
                     and lease["lease_digest"] == canonical_digest(lease, digest_field="lease_digest")
                     and all(lease[key] == entry[key] for key in ("lane", "name", "owner"))
                     and lease.get("consumer_lifetime_contract") == scratch.CONSUMER_LIFETIME_PROTOCOL
                     and lease["released_at_epoch"] is None and _epoch(lease["expires_at_epoch"])
                     and issued < lease["expires_at_epoch"], "experiment_consumer_inactive")
            result = cls()
            result.files, result.fd, result._authority_lock = files, parent, lock
            result.root, result.lane, result.name = root, entry["lane"], target.name
            result.exclusive, result.now = False, now
            result.entry, result.birth, result._head_record, result._lease_record = entry, birth, head_record, lease_record
            result._birth_record, result._projection_expiry = birth_record, authority["expires_at_epoch"]
            result._restoration_record = restoration_record
            result._started, result._checks = time.monotonic(), 0
            result._context_tokens = []
            result.identity = dict(root=str(root), lane=entry["lane"], name=target.name, owner=lease["owner"],
                run_ref=lease["run_ref"], lease_digest=lease["lease_digest"], consumer_lifetime_contract=scratch.CONSUMER_LIFETIME_PROTOCOL)
            result._producer_requests = []
            result._producer_bootstrap_record = None
            result._arena_selection_record = arena_selection_record
            result._producer_sources = ()
            result._public_root = AUTHORITY_ROOT
            result._producer_config_path = PRODUCER_CONFIG_PATH if _producer_config_path is None else _producer_config_path
            if _producer_bootstrap_path is not None:
                _require(_producer_request_paths is None and _producer_config_path is None,
                         "experiment_producer_authority_required")
                _admit_public_producer(result, files, target, public, _producer_bootstrap_path, gid, issued)
            elif _producer_request_paths is not None:
                _require(os.geteuid() == 0 and isinstance(_producer_request_paths, (list, tuple))
                         and len(_producer_request_paths) == 2 and birth["class_intent"] == "evidence"
                         and birth["cleanup"] == "owner_review"
                         and birth["writer_scope"] == "native_g1_development_pair.v1"
                         and birth["participant_profile"] in ("g1_local_prelaunch_block.v1", "g1_local_contained_completed.v1"),
                         "experiment_producer_authority_required")
                marker_raw, _ = files.read(target / ".registered-experiment.v1.json", cap=4096)
                _require(len(marker_raw) == birth["marker"]["size_bytes"]
                         and retained._digest(marker_raw, _work_budget=files.budget) == birth["marker"]["sha256"],
                         "experiment_birth_changed")
                marker = retained._document(marker_raw, 4096, _work_budget=files.budget)
                _require(marker["marker_digest"] == canonical_digest(marker, digest_field="marker_digest")
                         and marker["generation"] == entry["generation"], "experiment_birth_changed")
                private = AUTHORITY_ROOT.parent / "requests/experiment-records" / (entry["intent_id"] + ".json")
                intent_raw, _ = files.read(private, cap=32768, protected=True, mode=0o600)
                _require(len(intent_raw) == marker["intent"]["size_bytes"]
                         and retained._digest(intent_raw, _work_budget=files.budget) == marker["intent"]["sha256"],
                         "experiment_producer_authority_required")
                intent = retained._document(intent_raw, 32768, _work_budget=files.budget)
                _require(intent["intent_digest"] == canonical_digest(intent, digest_field="intent_digest")
                         and intent["generation"] == entry["generation"] and intent["intent_id"] == entry["intent_id"]
                         and isinstance(intent["request_records"], list) and len(intent["request_records"]) == 2,
                         "experiment_producer_authority_required")
                for path, selector in zip(_producer_request_paths, intent["request_records"], strict=True):
                    _require(isinstance(path, Path) and path.is_absolute(), "experiment_producer_request_changed")
                    request_raw, request_record = files.read(path, cap=65536, protected=True)
                    owners._identity(request_raw, selector["sha256"], selector["size_bytes"], files.budget)
                    request = retained._document(request_raw, 65536, _work_budget=files.budget)
                    _require(request["request_digest"] == canonical_digest(request, digest_field="request_digest"),
                             "experiment_producer_request_changed")
                    result._producer_requests.append((path, request_record, request["candidate_id"], request["request_digest"]))
            files.budget.close()
            result.check()
            accepted = True
            return result
        except (OSError, StopIteration, KeyError, TypeError):
            raise OwnerTargetVersionError("experiment_consumer_admission_failed") from None
        finally:
            if not accepted:
                if result is not None:
                    result._closed = True
                try:
                    files.finish()
                finally:
                    files.budget.close()

    def __enter__(self):
        self.check()
        _require(len(self._context_tokens) < 64, "experiment_consumer_resource_exhausted")
        self._context_tokens.append(_CURRENT_USE.set(self))
        return self

    def __exit__(self, *_exc):
        _require(bool(self._context_tokens), "experiment_consumer_scope_invalid")
        _CURRENT_USE.reset(self._context_tokens.pop())
        self.close()

    def authorize_g1_pair(self, paths):
        _require(isinstance(paths, (list, tuple)) and len(paths) == len(self._producer_requests) == 2,
                 "experiment_producer_authority_required")
        self.check()
        for path, supplied in zip(paths, self._producer_requests, strict=True):
            expected, record, _, _ = supplied
            _require(path == expected, "experiment_producer_request_changed")
            self.files.location(record.parent, cleanup=True)
            self.files.proof(record.fd)
            _require(owners._metadata(os.fstat(record.fd)) == owners._metadata(record.info)
                     == owners._metadata(os.stat(record.name, dir_fd=record.parent, follow_symlinks=False)),
                     "experiment_producer_request_changed")

    def authorize_worker(self, request, output):
        _require(any(output.name == candidate and request.get("request_digest") == digest
                     and canonical_digest(request, digest_field="request_digest") == digest
                     for _, _, candidate, digest in self._producer_requests), "experiment_producer_request_changed")
        self.check()

    def check(self):
        _require(not self._closed and self._checks < 20000 and time.monotonic() - self._started <= 4 * 3600,
                 "experiment_consumer_resource_exhausted")
        self._checks += 1
        issued = self.now()
        _require(_epoch(issued) and issued < min(self.entry["expires_at_epoch"], self._projection_expiry),
                 "experiment_consumer_inactive")
        files = self.files
        try:
            files.location(self.fd, cleanup=True)
            files.proof(self._authority_lock)
            fcntl.flock(self._authority_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            try:
                records = (self._head_record, self._lease_record, self._birth_record)
                if self._arena_selection_record is not None:
                    records += (self._arena_selection_record,)
                if self._producer_bootstrap_record is not None:
                    records += (self._producer_bootstrap_record,) + self._producer_sources
                if self._restoration_record is not None:
                    records += (self._restoration_record,)
                for record in records:
                    files.location(record.parent, cleanup=True)
                    files.proof(record.fd)
                    _require(owners._metadata(os.fstat(record.fd)) == owners._metadata(record.info)
                             == owners._metadata(os.stat(record.name, dir_fd=record.parent, follow_symlinks=False)),
                             "experiment_consumer_version_changed")
            finally:
                files.proof(self._authority_lock)
                fcntl.flock(self._authority_lock, fcntl.LOCK_UN)
        except OSError:
            raise OwnerTargetVersionError("experiment_consumer_check_failed") from None

    @contextmanager
    def borrow(self, output):
        _require(isinstance(output, Path) and output.parent == self.path, "experiment_worker_descendant_invalid")
        self.check()
        try:
            yield self
        finally:
            self.check()

    def refresh(self):
        raise OwnerTargetVersionError("experiment_explicit_new_admission_required")

    def mkdir(self, relative):
        raise OwnerTargetVersionError("experiment_consumer_operation_unsupported")

    def close(self):
        self._closed = True
        self.files.finish()

    def _take(self, fd):
        raise OwnerTargetVersionError("experiment_descriptor_ownership_unproven")

    def _take_all(self, descriptors):
        raise OwnerTargetVersionError("experiment_descriptor_ownership_unproven")

    def _open(self, *args, **kwargs):
        raise OwnerTargetVersionError("experiment_consumer_operation_unsupported")

    def _dup(self, original):
        raise OwnerTargetVersionError("experiment_consumer_operation_unsupported")


def _admit_public_producer(result, files, target, public, selected, gid, issued):
    """Retain root-selected readable inputs without opening private authority."""
    from .native_g1_registered_containment import SOURCE_MODULES
    entry, birth = result.entry, result.birth
    _require(isinstance(selected, Path) and selected == AUTHORITY_ROOT / (entry["intent_id"] + ".producer-bootstrap.json")
             and birth["participant_profile"] == "g1_local_contained_completed.v1"
             and birth["class_intent"] == "evidence" and birth["cleanup"] == "owner_review"
             and birth["writer_scope"] == "native_g1_development_pair.v1", "experiment_producer_authority_required")
    raw, original = _read(files, public, selected.name, 32768, gid)
    bootstrap = retained._document(raw, 32768, _work_budget=files.budget)
    fields = {"schema_version", "intent_id", "generation", "birth", "target_identity", "lease", "participant_profile",
              "intent", "requests", "installed_sources", "issued_at_epoch", "expires_at_epoch", "bootstrap_digest"}
    _require(type(bootstrap) is dict and set(bootstrap) == fields
             and bootstrap["schema_version"] == "control_plane_lane_experiment_producer_bootstrap.v1"
             and bootstrap["bootstrap_digest"] == canonical_digest(bootstrap, digest_field="bootstrap_digest")
             and all(bootstrap[key] == entry[key] for key in ("intent_id", "generation", "birth", "target_identity", "lease"))
             and bootstrap["participant_profile"] == birth["participant_profile"]
             and _epoch(bootstrap["issued_at_epoch"]) and _epoch(bootstrap["expires_at_epoch"])
             and bootstrap["issued_at_epoch"] <= issued < bootstrap["expires_at_epoch"] <= entry["expires_at_epoch"]
             and type(bootstrap["requests"]) is list and len(bootstrap["requests"]) == 2
             and type(bootstrap["installed_sources"]) is dict and set(bootstrap["installed_sources"]) == SOURCE_MODULES,
             "experiment_producer_authority_required")
    marker_raw, marker_record = files.read(target / ".registered-experiment.v1.json", cap=4096)
    owners._identity(marker_raw, birth["marker"]["sha256"], birth["marker"]["size_bytes"], files.budget)
    marker = retained._document(marker_raw, 4096, _work_budget=files.budget)
    _require(marker["marker_digest"] == canonical_digest(marker, digest_field="marker_digest")
             and marker["generation"] == entry["generation"] and marker["intent"] == bootstrap["intent"],
             "experiment_birth_changed")
    sources = [marker_record]
    for name in sorted(SOURCE_MODULES):
        source_raw, source_record = files.read(Path(__file__).parent / (name + ".py"), cap=1024 * 1024, protected=True)
        _require(_valid_digest(bootstrap["installed_sources"][name])
                 and retained._digest(source_raw, _work_budget=files.budget) == bootstrap["installed_sources"][name],
                 "experiment_producer_source_changed")
        sources.append(source_record)
    seen_paths, seen_candidates = set(), set()
    for row in bootstrap["requests"]:
        _require(type(row) is dict and set(row) == {"path", "raw", "candidate_id", "request_digest"}
                 and type(row["path"]) is str and owners._matches(row["candidate_id"], owners._OWNER)
                 and _valid_digest(row["request_digest"]), "experiment_producer_request_changed")
        path = retained._path(row["path"], _work_budget=files.budget)
        _require(path.is_absolute() and not path.is_relative_to(target) and path not in seen_paths
                 and row["candidate_id"] not in seen_candidates, "experiment_producer_request_changed")
        selector = row["raw"]
        _require(type(selector) is dict and set(selector) == {"sha256", "size_bytes"}, "experiment_producer_request_changed")
        request_raw, request_record = files.read(path, cap=65536, protected=True)
        _require(request_record.info.st_uid == 0 and request_record.info.st_gid == gid
                 and request_record.info.st_nlink == 1 and stat.S_IMODE(request_record.info.st_mode) == 0o640,
                 "experiment_producer_request_changed")
        owners._identity(request_raw, selector["sha256"], selector["size_bytes"], files.budget)
        request = retained._document(request_raw, 65536, _work_budget=files.budget)
        _require(request["candidate_id"] == row["candidate_id"]
                 and request["request_digest"] == row["request_digest"]
                 == canonical_digest(request, digest_field="request_digest"), "experiment_producer_request_changed")
        result._producer_requests.append((path, request_record, row["candidate_id"], row["request_digest"]))
        seen_paths.add(path)
        seen_candidates.add(row["candidate_id"])
    result._producer_sources = tuple(sources)
    result._producer_bootstrap_record = original
    result._projection_expiry = min(result._projection_expiry, bootstrap["expires_at_epoch"])


@contextmanager
def registered_reader(path):
    """Finite real reader scope; ambient use is rechecked, never a grant."""
    budget = ReferenceCollectionBudget(values_limit=10000)
    try:
        selected = Path(path)
        target = registered_target(selected, LANE_ROOTS, _budget=budget)
        if target is None:
            budget.close()
            yield None
            return
        current = _CURRENT_USE.get()
        if type(current) is RegisteredExperimentUse and current.path == target:
            budget.close()
            current.check()
            try:
                yield current
            finally:
                current.check()
        else:
            with RegisteredExperimentUse.admit(target, _metadata_budget=budget) as use:
                yield use
    finally:
        budget.close()
