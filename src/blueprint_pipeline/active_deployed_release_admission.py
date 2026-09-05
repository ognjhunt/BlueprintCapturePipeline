"""Offline checkout authority from the root-installed, currently active release."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ACTIVE_LINK = Path('/opt/blueprint/task-evaluation-control-plane')
RELEASES_ROOT = Path('/opt/blueprint/task-evaluation-control-plane-releases')
STATE_ROOT = Path('/var/lib/blueprint/pipeline-control-plane')
RECEIPTS_ROOT = STATE_ROOT / 'deploy-receipts'
SOURCE_CHECKOUT = Path('/opt/blueprint/BlueprintCapturePipeline')
TRUSTED_UID = 0
_COMMIT = re.compile(r'[0-9a-f]{40}')


def _trusted(path: Path, *, directory: bool = False) -> bool:
    try:
        return (not any(p.is_symlink() for p in (path, *path.parents))
                and (path.is_dir() if directory else path.is_file())
                and path.stat().st_uid == TRUSTED_UID and not path.stat().st_mode & 0o022)
    except OSError:
        return False


def _read(path: Path) -> tuple[dict, str]:
    if not _trusted(path) or path.stat().st_size > 1024 * 1024:
        raise ValueError('untrusted_release_evidence')
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('invalid_release_evidence')
    return value, 'sha256:' + hashlib.sha256(raw).hexdigest()


def _provenance(binding: dict, commit: str) -> tuple[dict, str]:
    if not isinstance(binding, dict):
        raise ValueError('provenance_binding_invalid')
    path = Path(str(binding.get('path') or ''))
    permitted = (STATE_ROOT / commit / 'deploy-release-provenance.json',
                 STATE_ROOT / 'task-evaluation-control-plane-releases' / commit / 'deploy-release-provenance.json')
    if path not in permitted:
        raise ValueError('provenance_path_invalid')
    value, digest = _read(path)
    if (binding.get('sha256') != digest or binding.get('size_bytes') != path.stat().st_size
            or binding.get('git_sha') != commit or value.get('git_sha') != commit
            or value.get('schema_version') != 'blueprint.deploy_release_provenance.v1'):
        raise ValueError('provenance_binding_invalid')
    claim = value.get('claim_boundary', {})
    if not isinstance(claim, dict):
        raise ValueError('provenance_claim_invalid')
    if value.get('status') == 'iteration':
        if not (set(value) == {'schema_version', 'status', 'git_sha', 'promotion_eligible', 'claim_boundary'}
                and value.get('promotion_eligible') is False
                and set(claim) == {'canonical_full_lane_verified', 'promotion_eligible', 'evidence_grade'}
                and claim.get('canonical_full_lane_verified') is False
                and claim.get('promotion_eligible') is False and claim.get('evidence_grade') == 'development_only'
                and binding.get('provenance_status') == 'iteration'
                and binding.get('promotion_eligible') is False
                and binding.get('canonical_full_lane_verified') is False
                and binding.get('run_id') is None and binding.get('run_url') is None):
            raise ValueError('iteration_provenance_invalid')
    elif not (value.get('status') == binding.get('provenance_status') == 'verified'
              and binding.get('promotion_eligible') is True
              and binding.get('canonical_full_lane_verified') is True
              and value.get('workflow_name') == 'Full Test Lane'
              and value.get('workflow_path') == '.github/workflows/full-test-lane.yml'
              and value.get('job_name') == 'Full pytest lane on CPU runner'
              and claim.get('canonical_full_lane_verified') is True
              and type(value.get('run_id')) is int and value['run_id'] > 0
              and binding.get('run_id') == value['run_id']
              and type(value.get('collection', {}).get('test_count')) is int
              and value['collection']['test_count'] > 0
              and value['collection'].get('skipped_count') == 0):
        raise ValueError('promotion_provenance_invalid')
    return value, digest


def inspect_active_deployed_release(repo_root: Path, commit: str) -> dict[str, Any] | None:
    """None is a standalone checkout; production checkouts must prove current activation.

    No environment variable or caller boolean supplies trust roots or authority.
    Iteration admission never grants production-promotion eligibility.
    """
    root = repo_root.resolve()
    if not root.is_relative_to(RELEASES_ROOT):
        return None
    blocked = {'status': 'blocked', 'blockers': ['gpu_canary_deployed_release_inactive_or_untrusted']}
    try:
        link_stat = ACTIVE_LINK.lstat()
        if not (_COMMIT.fullmatch(commit) and root == RELEASES_ROOT / commit
                and _trusted(RELEASES_ROOT, directory=True) and _trusted(root, directory=True)
                and _trusted(ACTIVE_LINK.parent, directory=True)
                and ACTIVE_LINK.is_symlink() and link_stat.st_uid == TRUSTED_UID
                and ACTIVE_LINK.resolve(strict=True) == root
                and _trusted(RECEIPTS_ROOT, directory=True)):
            return blocked
        blocked['blockers'] = ['gpu_canary_deployed_release_receipt_unverified']
        for receipt_path in sorted(RECEIPTS_ROOT.glob('*.json')):
            try:
                receipt, digest = _read(receipt_path)
                if receipt.get('source_commit') != commit:
                    continue
                intake = receipt.get('intake_runtime', {})
                surfaces = receipt.get('surfaces', [])
                if not isinstance(intake, dict) or not isinstance(surfaces, list):
                    continue
                if not (receipt.get('schema_version') == 'control_plane_commit_deploy_receipt.v1'
                        and receipt.get('status') == 'deployed' and receipt.get('release_path') == str(root)
                        and intake.get('source_commit') == commit and intake.get('commit_proven') is True
                        and {'name': 'active_release', 'head': commit, 'path': str(root)} in surfaces
                        and {'name': 'source_checkout', 'head': commit, 'path': str(SOURCE_CHECKOUT)} in surfaces):
                    continue
                provenance, provenance_digest = _provenance(receipt.get('release_provenance', {}), commit)
                current = ACTIVE_LINK.lstat()
                if ((current.st_ino, current.st_mtime_ns) != (link_stat.st_ino, link_stat.st_mtime_ns)
                        or ACTIVE_LINK.resolve(strict=True) != root):
                    return {'status': 'blocked', 'blockers': ['gpu_canary_deployed_release_changed_during_admission']}
                promoted = provenance['status'] == 'verified'
                return {'status': 'verified_active_release', 'blockers': [], 'source_commit': commit,
                        'receipt_path': str(receipt_path), 'receipt_sha256': digest,
                        'provenance_sha256': provenance_digest, 'promotion_eligible': promoted,
                        'release_admission_mode': 'promoted' if promoted else 'development_iteration',
                        'evidence_grade': 'production_promoted' if promoted else 'development_only'}
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                continue
    except (OSError, ValueError, TypeError):
        pass
    return blocked
