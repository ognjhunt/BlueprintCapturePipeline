# Selected evaluation CPU compilation replay

ADP-009D / day-21: replaying the real website-selected evaluation before deployment exposed `rigid_relocation_surface_target_binding_mismatch`. The retained scene/definition/success records use a Python-owned target digest with `stable_seconds: 1.0`; website transport deliberately uses a cross-runtime digest and represents it as `1`. Every physical and scoring field is the same.

The adapter now validates each original target seal, normalizes only for comparison through the existing cross-runtime target helper, and retains the native target in compiled evidence. Changed geometry/scoring or an invalid digest still refuses compilation. No source artifact is rewritten.

The same batch skips the new-call credential preflight only after a completed-placement adoption has passed intent validation. Such a continuation has zero new inference authority and reuses the original model/cost evidence. Fresh placements still require the early credential check.

Validation: 71 focused adapter, target, autostart and retained-placement tests pass; Ruff and source governance pass. Saved-input preparation reaches episode compilation in scratch with no paid execution. The original compilation refusal was observed before deployment; successful compilation and native execution remain separate proof requirements.
