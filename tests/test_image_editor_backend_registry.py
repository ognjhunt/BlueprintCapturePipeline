"""Adopting a newer image editor should be a data change, not a release.

The admissible set was a frozenset of three literals in the bundle module. The
best image-editing models change every few months, so a pipeline that needs a
code change to follow them stops following them.

Opening that seam must not open a second one. These are third-party models with
genuinely different terms and the artifacts reach customers, so the terms
travel with the name and an entry missing them is refused rather than defaulted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.image_editor_backend_registry import (
    ARTIFIXER_DIRECT_CAPABILITY,
    DEFAULT_REGISTRY_PATH,
    REGISTRY_SCHEMA_VERSION,
    REQUIRED_FIELDS,
    SEMANTIC_TEACHER_IMAGE_EDIT_CAPABILITY,
    ImageEditorRegistryError,
    admissible_for_delivery,
    load_registry,
    registered_backend_ids,
)


def _registry(tmp_path: Path, backends: list[dict]) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps({"schema_version": REGISTRY_SCHEMA_VERSION, "backends": backends}),
        encoding="utf-8",
    )
    return path


def _entry(**overrides) -> dict:
    row = {
        "backend_id": "new_sota_editor",
        "capability": ARTIFIXER_DIRECT_CAPABILITY,
        "model_identity": "Some Newer Editor v2",
        "license": "Apache-2.0",
        "license_url": "https://example.invalid/license",
        "commercial_use_permitted": True,
        "recorded_on": "2026-08-13",
    }
    row.update(overrides)
    return row


def test_the_shipped_registry_loads_and_covers_the_backends_in_use() -> None:
    registry = load_registry()

    assert DEFAULT_REGISTRY_PATH.is_file()
    assert {"artifixer", "vibe_image_edit"} <= set(registry)


def test_gpt_image_2_omits_the_unsupported_input_fidelity_parameter() -> None:
    backend = load_registry()["openai_gpt_image_2_2026_04_21_semantic_teacher"]
    execution = backend["execution"]

    assert execution["high_fidelity_input_supported"] is True
    assert execution["input_fidelity_parameter_supported"] is False
    assert "input_fidelity" not in execution["default_options"]
    assert canonical_digest(backend) == (
        "sha256:fd4669469e0d4f8155acb6687824817ce13147a39bb5f417734a987584b69fb7"
    )


def test_the_bundle_takes_its_admissible_set_from_the_registry() -> None:
    """The point of the change: no literal set in the consuming module."""

    from blueprint_pipeline import public_scene_artifixer3d_bundle as bundle

    assert bundle.DIRECT_EDITOR_BACKENDS == registered_backend_ids(
        capability=ARTIFIXER_DIRECT_CAPABILITY
    )
    source = Path(bundle.__file__).read_text(encoding="utf-8")
    assert 'frozenset({"artifixer"' not in source, "the literal set came back"


def test_a_new_backend_is_admitted_by_adding_a_row(tmp_path: Path) -> None:
    """Adopting next quarter's model should not touch any Python."""

    path = _registry(tmp_path, [_entry()])

    assert registered_backend_ids(path) == frozenset({"new_sota_editor"})
    assert admissible_for_delivery("new_sota_editor", path=path) is True


def test_capabilities_keep_semantic_teacher_rows_out_of_direct_editor_lane(
    tmp_path: Path,
) -> None:
    semantic = _entry(
        backend_id="semantic_only",
        capability=SEMANTIC_TEACHER_IMAGE_EDIT_CAPABILITY,
    )
    direct = _entry()
    path = _registry(tmp_path, [direct, semantic])

    assert registered_backend_ids(
        path, capability=ARTIFIXER_DIRECT_CAPABILITY
    ) == frozenset({"new_sota_editor"})
    assert registered_backend_ids(
        path, capability=SEMANTIC_TEACHER_IMAGE_EDIT_CAPABILITY
    ) == frozenset({"semantic_only"})


def test_unknown_capability_is_refused(tmp_path: Path) -> None:
    path = _registry(tmp_path, [_entry(capability="wrong_lane")])

    with pytest.raises(ImageEditorRegistryError, match="capability_invalid"):
        load_registry(path)


@pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
def test_an_entry_missing_any_required_field_is_refused(tmp_path: Path, field: str) -> None:
    path = _registry(tmp_path, [_entry(**{field: None})])

    with pytest.raises(ImageEditorRegistryError) as excinfo:
        load_registry(path)

    assert "incomplete" in str(excinfo.value) or "terms_unrecorded" in str(excinfo.value)


def test_unrecorded_terms_are_refused_rather_than_assumed(tmp_path: Path) -> None:
    """A string here would read as permission without anyone granting it."""

    path = _registry(tmp_path, [_entry(commercial_use_permitted="probably fine")])

    with pytest.raises(ImageEditorRegistryError) as excinfo:
        load_registry(path)

    assert "terms_unrecorded" in str(excinfo.value)


def test_a_non_commercial_backend_is_registered_but_not_deliverable(tmp_path: Path) -> None:
    """Registered means usable for research, not automatically for customers."""

    path = _registry(tmp_path, [_entry(commercial_use_permitted=False)])

    assert "new_sota_editor" in registered_backend_ids(path)
    assert admissible_for_delivery("new_sota_editor", path=path) is False


def test_an_unregistered_backend_refuses_rather_than_returning_false(tmp_path: Path) -> None:
    """"Not allowed" and "never checked" must not look alike to a caller."""

    path = _registry(tmp_path, [_entry()])

    with pytest.raises(ImageEditorRegistryError) as excinfo:
        admissible_for_delivery("some_model_nobody_registered", path=path)

    assert "unregistered" in str(excinfo.value)


def test_the_reserved_none_value_cannot_be_registered(tmp_path: Path) -> None:
    """`none` means no editor ran; a backend named `none` would be ambiguous."""

    path = _registry(tmp_path, [_entry(backend_id="none")])

    with pytest.raises(ImageEditorRegistryError) as excinfo:
        load_registry(path)

    assert "reserved" in str(excinfo.value)


def test_a_duplicate_backend_id_is_refused(tmp_path: Path) -> None:
    path = _registry(tmp_path, [_entry(), _entry(license="MIT")])

    with pytest.raises(ImageEditorRegistryError) as excinfo:
        load_registry(path)

    assert "duplicate" in str(excinfo.value)


def test_an_empty_registry_is_refused(tmp_path: Path) -> None:
    """Silently admitting nothing would read as "no editor is allowed"."""

    with pytest.raises(ImageEditorRegistryError):
        load_registry(_registry(tmp_path, []))


def test_a_missing_registry_refuses_rather_than_defaulting(tmp_path: Path) -> None:
    with pytest.raises(ImageEditorRegistryError) as excinfo:
        load_registry(tmp_path / "absent.json")

    assert "unreadable" in str(excinfo.value)
