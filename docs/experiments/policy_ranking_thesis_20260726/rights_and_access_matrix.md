# Rights and access matrix

Snapshot: 2026-07-26. This is an engineering access record, not legal advice.

| Component | Pinned source | Stated terms | Current use | Restriction carried forward |
|---|---|---|---|---|
| RoboArena DataDump 07-17-2026 | `7931db81f3f6a48a3245427f7213a4c461f92ccc` | MIT | Frozen calibration metadata and selected action files | PII fields are excluded; labels remain sealed until their registered partition is scored |
| OSCAR code | `4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb` | Apache-2.0 | Reproducibility reference | Code license does not establish rights to every released rollout video |
| OSCAR-2B model | `c9781ffa7dd8556d862d7d9f338a2ea008a58ca6` | Apache-2.0 | Pinned future reproducibility path | Model has not been run in this goal yet |
| OSCAR policy rollouts | `db5edfaef285c15d0a41d5115177a983c08b4f5f` | No explicit dataset license metadata found | Internal, label-blind evaluation only | Do not redistribute or use for customer deliverables without rights clearance |
| OpenPI code | `15a9616a00943ada6c20a0f158e3adb39df2ccac` | Apache-2.0 | Exact DROID policy configs and adapter preflight | Code license alone does not establish checkpoint-specific terms |
| Four OpenPI RoboArena PaliGemma checkpoints | Public `gs://openpi-assets` objects; manifests pinned in `captured_site_policy_preflight.json` | No checkpoint-specific license metadata verified | Attributable prospective candidates; not downloaded yet | Internal experiment only until checkpoint terms are verified |
| Ctrl-World code | `99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d` | MIT | Comparator design and adapter audit | Its released replay interface expects normalized 7-D Cartesian EEF state; arbitrary joint NPZ replay must fail closed |
| Ctrl-World checkpoint | `8cf814` model revision recorded in research notes | Stable Video Diffusion community license dependency | Not downloaded or run | Commercial conditions and action conversion require review before use |
| NVIDIA SimReady Warehouse | `c7fe115cb79c7ddbd0532630d7768b5736b0ecc4` | CC-BY-4.0 | Controlled-scene specification; one spray-can dependency tree materialized | Full 15 GB dependency archive is not downloaded; warehouse is a diagnostic, never a physical answer key |
| Voxel51 playroom 3DGS | `6aaba1b1dd0439c0db464842f8a6600a37e0eae9` | Apache-2.0 | Public captured-site visual layer and local preview | Source lacks camera/metric calibration sidecars needed to prove metric registration |
| MuJoCo Menagerie Franka | `71f066ad0be9cd271f7ed58c030243ef157af9f4` | Apache-2.0 | Pinned visual/kinematic robot asset | Current preview uses simpler geometry proxies; it is not an articulated-asset execution |
| InteriorGS | Not accessed | Custom/gated terms | None | Access terms were not accepted and no files were downloaded in this goal |
| OpenAI GPT-5 judge | `gpt-5-2025-08-07` | API terms and metered pricing | Request inventory only; zero calls so far | Requires explicit user approval; generated OSCAR crop only, `store=false`, labels/PII/physical half forbidden |

The experiment can proceed internally with the public, pinned components above. Any rollout-media redistribution, customer-facing use, gated InteriorGS access, or new paid provider lane remains separately gated.
