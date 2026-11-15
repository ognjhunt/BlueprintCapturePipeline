"""G52: whether a site has withdrawn is read from the configured roots only, and an unreadable bucket is unknown."""
from __future__ import annotations

from blueprint_pipeline import screening_site_withdrawal as withdrawal


def _withdraw(root, site="site-job-1", bucket=None):
    site_dir = (root / bucket if bucket else root) / "scenes" / site / "website_withdrawal"
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "tombstone.json").write_text("{}")


def _roots(monkeypatch, tmp_path):
    roots = tmp_path / "handoffs"
    roots.mkdir()
    monkeypatch.setenv(withdrawal.SITE_ROOTS_ENV, str(roots))
    return roots


def test_state_reads_only_configured_roots(tmp_path, monkeypatch):
    monkeypatch.delenv(withdrawal.SITE_ROOTS_ENV, raising=False)
    assert withdrawal.site_withdrawal_state("site-job-1") == "not_configured"
    roots = _roots(monkeypatch, tmp_path)
    assert withdrawal.site_withdrawal_state("site-job-1") == "not_withdrawn"
    _withdraw(roots, site="site-job-2")
    assert withdrawal.site_withdrawal_state("site-job-2") == "withdrawn"
    assert withdrawal.site_withdrawal_state("site-job-1") == "not_withdrawn"


def test_a_hidden_folder_is_never_a_bucket(tmp_path, monkeypatch):
    # The storage clean-up's root-only `.scene-workspace-inventory` sits beside the buckets; the screening unit runs
    # as `blueprint`, and a look inside it raised PermissionError. The deletion scan already skips hidden folders.
    import pathlib
    roots = _roots(monkeypatch, tmp_path)
    _withdraw(roots, site="site-job-2", bucket=".website_preparation_pending")
    hidden = roots / ".scene-workspace-inventory"
    hidden.mkdir()
    real_stat = pathlib.Path.stat

    def stat(self, *args, **kwargs):
        if self.is_relative_to(hidden):
            raise PermissionError(13, "Permission denied", str(self))
        return real_stat(self, *args, **kwargs)
    monkeypatch.setattr(pathlib.Path, "stat", stat)
    assert withdrawal.site_withdrawal_state("site-job-1") == "not_withdrawn"
    assert withdrawal.site_withdrawal_state("site-job-2") == "not_withdrawn"
    _withdraw(roots, site="site-job-2", bucket="blueprint-8c1ca.appspot.com")
    assert withdrawal.site_withdrawal_state("site-job-2") == "withdrawn"


def _unreadable(monkeypatch, folder):
    """Every look inside ``folder`` raises PermissionError, as a folder the screening user may not search does."""
    import pathlib
    real_stat = pathlib.Path.stat

    def stat(self, *args, **kwargs):
        if self.is_relative_to(folder):
            raise PermissionError(13, "Permission denied", str(self))
        return real_stat(self, *args, **kwargs)
    monkeypatch.setattr(pathlib.Path, "stat", stat)


def test_a_bucket_that_cannot_be_read_is_unknown_never_not_withdrawn(tmp_path, monkeypatch):
    roots = _roots(monkeypatch, tmp_path)
    (roots / "_pipeline_failures").mkdir()
    _unreadable(monkeypatch, roots / "_pipeline_failures")
    assert withdrawal.site_withdrawal_state("site-job-1") == "unknown"
    _withdraw(roots, site="site-job-1", bucket="blueprint-8c1ca.appspot.com")
    assert withdrawal.site_withdrawal_state("site-job-1") == "withdrawn"
