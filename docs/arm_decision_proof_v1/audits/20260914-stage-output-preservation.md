# Preserve completed scene-configuration stages

ADP-009B/009D dependency for the public-scene rehearsal and day-14 two-candidate receipt gate.

Observed on Vast 50962751: ArtiFixer completed 30,000 steps and exported its model. Stage 3 subsequently failed. Final packaging then refused a normal shared-library symlink under `astra_cad_blender_runtime/packaged_blender`. The controller's SSH recovery required a successful ZIP marker and did not support the scene-configuration bundle kind. Automatic teardown consequently removed the only saved model checkpoint. The operator retained the 16 rendered images and review metadata, not the model weights. This run is not an end-to-end success.

The repair uses a shared output-archive writer that omits reproducible runtime installations, including packaged Blender, while retaining strict refusal of other symlinks. The stage chain atomically writes a completed-prefix checkpoint archive after each completed stage. A later checkpoint failure leaves the previous checkpoint intact. Partial checkpoints explicitly say the whole run is not completed.

Before teardown, scene-configuration recovery now attempts a bounded, pinned-SSH read even without a successful ZIP marker. If the final archive is missing or invalid, it tries the completed-stage checkpoint. Both attempts share the recovery deadline and retain byte-size, free-space, host-key and SHA checks. Recovering artifacts never grants successful execution or appearance qualification.

The same CAD/Blender bootstrap used by stage 3 now runs before the expensive stage chain. It checks packaged binaries, Python dependencies and the real asset sandbox without a model call. This moves detectable runtime refusals ahead of ArtiFixer training. Provider failures also emit their redacted cause to the container log independently of archive creation.

Focused checks cover the exact packaged-Blender symlink, retained trained-checkpoint bytes, strict refusal of unrelated symlinks, stage-3 failure after completed stages, preservation of an earlier checkpoint when a new snapshot fails, bounded pinned-SSH recovery, and early CAD refusal before training. Existing Astra driver/runtime tests exercise the shared bootstrap. Live deployment and the next runtime are separate evidence; these tests cannot restore the already lost checkpoint or establish that the unknown original stage-3 error is fixed.
