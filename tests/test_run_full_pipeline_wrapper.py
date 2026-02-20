"""Static guards for strict defaults in run_full_pipeline.sh."""

from __future__ import annotations

from pathlib import Path


def _script_text() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "scripts" / "run_full_pipeline.sh").read_text(encoding="utf-8")


def test_wrapper_defaults_to_full_required_completion_mode() -> None:
    text = _script_text()
    assert 'COMPLETION_MODE="${COMPLETION_MODE:-full_required}"' in text
    assert "--completion-mode MODE" in text


def test_wrapper_enforces_strict_env_in_full_mode() -> None:
    text = _script_text()
    assert "export PIPELINE_STANDALONE_MODE=false" in text
    assert "export RUNTIME_PREFLIGHT_ENABLED=true" in text
    assert "export ADVANCED_QUALITY_GATES_ENABLED=true" in text
    assert "export SAM3_TRACKING_MODE" in text
    assert "export SWAP_INCLUDE_HEURISTIC_AS_EXPLICIT=false" in text


def test_wrapper_validates_full_runtime_and_non_stub_scene() -> None:
    text = _script_text()
    assert "validate_full_runtime" in text
    assert "usd-assembly-job/assemble_scene.py" in text
    assert "scene.usda is a standalone stub" in text


def test_wrapper_uses_quality_first_nurec_defaults() -> None:
    text = _script_text()
    assert 'NUREC_QUALITY_PROFILE="${NUREC_QUALITY_PROFILE:-quality_first}"' in text
    assert 'MAX_FRAMES="${MAX_FRAMES:-320}"' in text
    assert 'EXTRACT_FPS="${EXTRACT_FPS:-5}"' in text
    assert 'N_ITERATIONS="${N_ITERATIONS:-9000}"' in text
    assert 'COLMAP_MATCHER_MODE="${COLMAP_MATCHER_MODE:-auto}"' in text
    assert 'COLMAP_SEQUENTIAL_OVERLAP="${COLMAP_SEQUENTIAL_OVERLAP:-30}"' in text
    assert 'COLMAP_CHUNKED_MODE="${COLMAP_CHUNKED_MODE:-auto}"' in text
    assert 'COLMAP_CHUNK_MIN_FRAMES="${COLMAP_CHUNK_MIN_FRAMES:-900}"' in text
    assert 'COLMAP_CHUNK_SIZE_FRAMES="${COLMAP_CHUNK_SIZE_FRAMES:-600}"' in text
    assert 'COLMAP_CHUNK_OVERLAP_FRAMES="${COLMAP_CHUNK_OVERLAP_FRAMES:-120}"' in text
    assert 'COLMAP_CHUNK_MAX_CHUNKS="${COLMAP_CHUNK_MAX_CHUNKS:-24}"' in text
    assert 'COLMAP_CHUNK_MATCHER_MODE="${COLMAP_CHUNK_MATCHER_MODE:-sequential}"' in text
    assert 'COLMAP_RETRY_MATCHER_MODE="${COLMAP_RETRY_MATCHER_MODE:-auto}"' in text
    assert 'NUREC_RESUME="${NUREC_RESUME:-false}"' in text


def test_wrapper_exposes_resume_and_fixer_rerun_flags() -> None:
    text = _script_text()
    assert "--resume                Enable NuRec resume mode" in text
    assert "--fixer-rerun           Force rerun of Fixer in resume mode" in text
    assert "--fixer-required        Fail if Fixer does not produce refined outputs" in text
    assert "--resume)          NUREC_RESUME=true;" in text
    assert "--fixer-rerun)     FIXER_RERUN=true;" in text
    assert "--fixer-required)  FIXER_REQUIRED=true;" in text
    assert 'NUREC_FIXER_ARGS+=(--fixer-rerun)' in text
    assert 'NUREC_FIXER_ARGS+=(--fixer-required)' in text


def test_wrapper_copies_object_index_before_rewrite() -> None:
    text = _script_text()
    assert 'cp -f "$INDEX_SOURCE" "$INDEX_POINTER_CANONICAL"' in text
    assert 'ln -sfn "object_point_cloud_index.json" "$INDEX_POINTER_LEGACY"' in text


def test_wrapper_uses_force_refresh_symlink_for_object_crops() -> None:
    text = _script_text()
    assert 'ln -sfn "${NUREC_OUTPUT_DIR}/object_crops" "${NUREC_ROOT}/object_crops"' in text


def test_wrapper_passes_chunked_colmap_flags_to_nurec_shim() -> None:
    text = _script_text()
    assert '--colmap-chunked-mode "$COLMAP_CHUNKED_MODE"' in text
    assert '--colmap-chunk-min-frames "$COLMAP_CHUNK_MIN_FRAMES"' in text
    assert '--colmap-chunk-size-frames "$COLMAP_CHUNK_SIZE_FRAMES"' in text
    assert '--colmap-chunk-overlap-frames "$COLMAP_CHUNK_OVERLAP_FRAMES"' in text
    assert '--colmap-chunk-max-chunks "$COLMAP_CHUNK_MAX_CHUNKS"' in text
    assert '--colmap-chunk-matcher-mode "$COLMAP_CHUNK_MATCHER_MODE"' in text
    assert '--colmap-retry-matcher-mode "$COLMAP_RETRY_MATCHER_MODE"' in text


def test_wrapper_has_max_n_gaussians_default_and_passthrough() -> None:
    text = _script_text()
    assert 'MAX_N_GAUSSIANS="${MAX_N_GAUSSIANS:-}"' in text
    assert "NUREC_GAUSSIAN_ARGS=()" in text
    assert 'if [ -n "${MAX_N_GAUSSIANS}" ]; then' in text
    assert 'NUREC_GAUSSIAN_ARGS+=(--max-n-gaussians "$MAX_N_GAUSSIANS")' in text
    assert '"${NUREC_GAUSSIAN_ARGS[@]}"' in text
