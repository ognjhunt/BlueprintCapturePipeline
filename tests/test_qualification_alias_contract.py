"""Contract tests for the deprecated ``blueprint_pipeline.qualification`` alias.

The capture->site-package orchestration spine moved from ``qualification.py``
to ``site_package_orchestrator.py`` (legacy naming misled agents about where
the product core lives; readiness/review outputs are the optional layer, the
orchestrator is not). These tests pin the compatibility contract: older
imports and dotted monkeypatch paths must keep resolving to the renamed
module object itself.
"""


def test_qualification_alias_is_site_package_orchestrator():
    import blueprint_pipeline.qualification as legacy
    from blueprint_pipeline import site_package_orchestrator

    assert legacy is site_package_orchestrator


def test_qualification_alias_from_import_resolves_same_symbols():
    from blueprint_pipeline import site_package_orchestrator
    from blueprint_pipeline.qualification import run_qualification_pipeline

    assert run_qualification_pipeline is site_package_orchestrator.run_qualification_pipeline


def test_qualification_alias_supports_dotted_monkeypatch(monkeypatch):
    from blueprint_pipeline import site_package_orchestrator

    sentinel = object()
    monkeypatch.setattr(
        "blueprint_pipeline.qualification._qualification_alias_probe",
        sentinel,
        raising=False,
    )
    assert site_package_orchestrator._qualification_alias_probe is sentinel
