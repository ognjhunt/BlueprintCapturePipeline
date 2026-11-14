"""One effective-catalog implementation: production screening and the intake preview agree.

Every registration here is a development-only fixture. The digests compared are
the catalog's own identity; none of them is evidence that a robot or policy ran.
"""
# Covers: src/blueprint_pipeline/beta_policy_effective_catalog.py
# Covers: src/blueprint_pipeline/beta_policy_screening_g1_catalog.py
# Covers: src/blueprint_pipeline/beta_policy_screening_files.py
# Covers: src/blueprint_pipeline/team_candidate_credential_file.py
# Covers: src/blueprint_pipeline/team_candidate_registry_reader.py
# Covers: src/blueprint_pipeline/team_candidate_rights.py
from __future__ import annotations



from blueprint_pipeline import beta_policy_effective_catalog as effective
from blueprint_pipeline import beta_policy_screening_team as team



def test_existing_entrypoints_reexport_the_single_shared_implementation():
    from blueprint_pipeline import beta_policy_screening_files as files
    from blueprint_pipeline import beta_policy_screening_g1 as g1
    from blueprint_pipeline import beta_policy_screening_g1_catalog as g1_catalog
    from blueprint_pipeline import beta_policy_screening_launch as launch
    from blueprint_pipeline import team_candidate_credential_file as credential
    from blueprint_pipeline import team_candidate_quick10_admission as admission
    from blueprint_pipeline import team_candidate_quick10_registry as registry
    from blueprint_pipeline import team_candidate_quick10_vast as vast
    from blueprint_pipeline import team_candidate_registry_reader as reader
    from blueprint_pipeline import team_candidate_rights as rights

    shared = {
        (team, effective): ("catalog_for_profile", "team_policy_interface", "_registration_rights"),
        (g1, g1_catalog): ("CANDIDATES", "catalog_for_profile"),
        (launch, files): ("_read", "_record", "_bound_path"),
        (vast, credential): ("credential_file_blockers",),
        (admission, rights): ("TeamCandidateAdmissionError", "RIGHTS_SCHEMA", "validate_candidate_rights",
                              "validate_provider_disclosure"),
        (registry, reader): ("TeamCandidateRegistryError", "REGISTRY_ENV", "REGISTRY_SCHEMA", "DERIVED_SCHEMA",
                             "DERIVED_FIELDS", "OPERATOR_FIELDS", "load_registry", "_read", "_operator_file",
                             "_derived_settings", "_derived_path", "_site_screening_profile"),
    }
    for (production, extracted), names in shared.items():
        for name in names:
            assert getattr(production, name) is getattr(extracted, name), (production.__name__, name)


