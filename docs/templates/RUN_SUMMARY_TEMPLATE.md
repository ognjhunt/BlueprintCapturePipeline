# Pipeline Run Summary Template

Use this template for every long run so results are comparable.

## Inputs
- input_video:
- scene_id:
- capture_id:
- workspace:
- gcs_root:
- bucket:

## Commit
- blueprint_capture_pipeline:
- blueprintpipeline:

## Params
- completion_mode:
- nurec_rerun_profile:
- nurec_quality_profile:
- reconstruction_backend:
- reconstruction_compare_backends:
- reconstruction_compare_winner:
- reconstruction_compare_report:
- skip_nurec:
- skip_fixer:
- skip_dense:
- fixer_mode:
- post_stage4_refine:
- post_stage4_refine_model:
- generation_provider_chain:
- scene_cleaning_mode:
- sam3_mask_export_space:
- inpaint360gs_resolution:
- pipeline_mode:
- colmap_min_registered_ratio:
- colmap_retry_min_registered_ratio:

## Outputs
- nurec_output_dir:
- pipeline_dir:
- scene_root:
- pipeline_root:
- scene_usda:
- swap_quality_report:
- pipeline_summary_json:
- orchestrator_run_report:
- reconstruction_compare_report:
- log_summary_json:
- log_summary_md:

## Runtime
- started_at_utc:
- ended_at_utc:
- duration_sec:
- status:
- exit_code:
- orchestrator_status:

## Failures
- none
