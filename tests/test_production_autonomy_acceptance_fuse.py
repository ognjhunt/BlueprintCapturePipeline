"""At ``fuse`` the acceptance takes a publication whose site limit passes $15 up to the job's ceiling; at
``nested`` (shipped) it refuses it exactly as before (``test_production_autonomy_acceptance``, unchanged)."""
import json

import pytest

from blueprint_pipeline import production_autonomy_acceptance as acceptance
from tests.test_production_autonomy_acceptance import replace_publication, report, retained  # noqa: F401


@pytest.mark.parametrize("retained", [{"publication_version": "v2"}], indirect=True)
@pytest.mark.parametrize("mode, bound, refused", [("nested", "15", True), ("fuse", "100", False)])
def test_a_publication_limit_above_15_follows_the_switch(retained, tmp_path, monkeypatch, mode, bound, refused):  # noqa: F811
    from blueprint_pipeline import website_job_limits

    path = tmp_path / "limits.json"
    path.write_text(json.dumps({"schema_version": website_job_limits.SCHEMA, "mode": mode}))
    monkeypatch.setenv(website_job_limits.CONFIG_ENV, str(path))
    # The fixture's plan has no job ceiling; stand in the bound a plan sized to a $100 ceiling gets at fuse.
    monkeypatch.setattr(acceptance, "publication_limit_bound",
                        lambda plan: acceptance.PUBLICATION_LIMIT_USD if mode == "nested" else acceptance.Decimal(bound))
    replace_publication(retained, lambda value: value["budget"].update(limit_usd="48"))
    assert ("publication_contract_invalid" in report(retained)["blockers"]) is refused
