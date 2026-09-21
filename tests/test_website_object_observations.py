"""ADP-030: asset appearance uses original capture detail, geometry keeps its grid."""
from pathlib import Path

from PIL import Image
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_object_observations import materialize_object_observations, validate_observation_handoff
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from tests.test_website_task_preparation import _arguments


def test_authoring_retains_original_pixels_and_rechecks_source_on_reuse(tmp_path):
    args = _arguments(tmp_path)
    geometry, masks = args['source_geometry'], args['task_masks']
    frame = geometry['frames'][0]
    original = tmp_path / 'original.png'
    Image.new('RGB', (1080, 1920), '#2244cc').save(original)
    frame.update(source_image_path=str(original), source_image_digest=_sha256_file(original))
    geometry['digest'] = canonical_digest(geometry, digest_field='digest')
    masks['source_geometry_digest'] = geometry['digest']
    masks['digest'] = canonical_digest(masks, digest_field='digest')
    preparation = compile_website_scene_preparation(**args)
    kwargs = dict(preparation=preparation, source_geometry=geometry, task_masks=masks, output_root=tmp_path / 'observations')
    result = materialize_object_observations(**kwargs)
    manifest, frames = validate_observation_handoff(Path(result['manifest']['path']), configuration=result['configuration'])
    assert frames[0].read_bytes() == original.read_bytes()
    assert manifest['frames'][0]['image_basis'] == 'original_capture'
    assert manifest['frames'][0]['triangle_count'] > 0
    assert materialize_object_observations(**kwargs) == result
    original.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source_changed'):
        materialize_object_observations(**kwargs)
