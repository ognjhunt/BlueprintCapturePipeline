# Merge validation — 2026-09-08 UTC

Integration base: `314fb214734e80f12bd00fd3ba7dc24cc91a8eb7`.
PR: https://github.com/ognjhunt/BlueprintCapturePipeline/pull/1768

The local audit branch was integrated in a separate worktree. Two conflicts were resolved by preserving main's media-integrity guard and retaining the added runtime modules. The initial integration episode-path run passed 206 tests. Both actual loopback transport tests passed; the GR00T fixture now serves the added modality recheck before the action request and has a bounded timeout.

The full CPU run https://github.com/ognjhunt/BlueprintCapturePipeline/actions/runs/34178062085 finished with 18,148 passing tests and 15 failures. One was introduced by this patch: the episode module exceeded its 2,000-line limit. Evidence/readiness helpers were extracted into the existing episode-evidence module, retaining the original error classes at the episode boundary. The episode module is now 1,991 lines; its focused tests and budget check passed (67 tests). No budget or guard was relaxed.

The other 14 failures were reproduced on an untouched detached checkout of the integration base, independently of this patch:

- `test_control_plane_storage_roots.py::test_every_root_named_by_a_production_unit_is_classified`
- `test_quality_gap_ledger.py::test_current_gap_ledger_maps_all_107_acceptance_criteria_and_derives_status`
- `test_release_quality_governance.py::test_repository_source_governance_policy_is_satisfied` (baseline ArtiFixer/Vast limits)
- `test_release_quality_governance.py::test_vast_adapter_stays_under_its_source_governance_budget`
- `test_materializer_reachability.py::test_unreachable_materializers_only_ever_decrease`
- `test_materializer_reachability.py::test_every_input_step_can_supply_its_materializer[semantic-teacher]`
- `test_control_plane_disk_budget.py::test_headroom_projects_refused_roles_without_paths`
- `test_task_evaluation_installed_source_bindings.py::test_public_preparation_entrypoint_loads_operator_binding_after_commit_validation`
- `test_task_evaluation_installed_source_bindings.py::test_s3_only_worker_request_ignores_stale_unrelated_installation`
- `test_live_pipeline_import_isolation.py::test_control_plane_lane_does_not_import_hot_lane`
- `test_live_pipeline_import_isolation.py::test_sam_phase_queue_stays_independent_of_execution_services`
- `test_production_runtime_env_guard_allocator_import.py::test_covers_every_module_the_systemd_units_execute`
- `test_task_evaluation_scene_configuration_diagnostic_builtin_integration.py::test_real_builtin_chain_accepts_checkpoint_hydration_and_skips_paid_prefix`
- `test_image_editor_backend_registry.py::test_gpt_image_2_omits_the_unsupported_input_fidelity_parameter`

The baseline probes finished with 2 failures in the first isolated run, then the remaining 12 failures and 47 passes in the expanded named-file run. They did not run a second full suite or alter main. The historical ledger was not rewritten, root retention classifications were not relaxed, and unrelated live-execution code was not changed merely to make these baseline failures green.

The required protected-branch context is `Impacted tests and sentinels`. The ordinary protected merge must wait for that check at the current PR head. The full-suite result is explicitly baseline-red; this merge is not a full-suite-green or deployment-readiness claim. Deployment and live execution remain with their existing owner.
