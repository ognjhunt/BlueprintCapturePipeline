# SPEC-10: Validate LLM enrichment output + add OSCAR-style clip captioning

- Status: Proposed
- Priority: **P1 — major**
- Area: `src/blueprint_pipeline/capture_enrichment_llm.py`, `object_index_stage.py`, `qualification.py`, new captioning stage
- Paper: OSCAR (arXiv 2606.04463) §captioning (VLM, 80–100 words, controlled fps)

## Problem

1. **Unvalidated LLM output enters labels/prompts.** `capture_enrichment_llm.py` defines
   a JSON schema (`_skill_schema`, `:69-145`) but enforces it only on the Codex-CLI path
   (`--output-schema`, `:210-217`). The Claude HTTP runner (`:282-328`) and OpenAI SDK
   runner (`:243-279`) return the model's parsed JSON if it is merely a `Mapping`
   (`:279`, `:328`). Consumers use it directly — object-index prompt-bank expansion
   (`object_index_stage.py:1504-1522`) and qualification enrichment
   (`qualification.py:4676-4703`) — so malformed or hallucinated fields flow unchecked
   into object labels and task hints.
2. **No clip captioning stage exists.** OSCAR's pipeline captions every curated clip with
   a VLM (80–100 words at controlled fps) because captions are the semantic index for
   training and retrieval. Our enrichment produces only prompt terms / relevance scores /
   task hints; Post-Training Data Packages ship robot-POV clips without grounded captions.
3. Default models in the enrichment config are dated (`gpt-5.1-mini`,
   `claude-3-7-sonnet-latest`, `:55`) and the provider defaults to `disabled` (`:44-66`),
   so where enrichment matters it is either off or unvalidated.

## Why this matters for launch

Fabricated labels are a rights/trust problem, not just a quality problem: a hallucinated
object label or task hint in a sold package is fake data under our doctrine. Missing
captions reduce package value directly — buyers use them for filtering, retrieval, and
conditioning (per the reference paper's design).

## Proposed fix

1. **Schema-validate every enrichment payload** regardless of provider: run
   `jsonschema.validate` against `_skill_schema` on the SDK/HTTP paths; on failure, retry
   once with the validation error in-prompt, then mark the enrichment
   `failed_validation` and exclude it (never partially ingest).
2. **Grounding checks:** enrichment outputs referencing objects/rooms must be
   cross-checked against the detected-object vocabulary of the bundle; out-of-vocabulary
   labels are flagged `unverified` and excluded from sold packages by default.
3. **Add a `clip_captioning_stage`** for curated clips (post SPEC-02/03): swappable VLM
   provider (consistent with the model-swap doctrine), 80–100-word target with word-count
   and language checks, frame-sampling rate configurable per clip length (OSCAR: 15 fps,
   1–2 fps for long episodes). Captions stored as derived artifacts with model
   name/version provenance and `generated: true` labeling.
4. Refresh default model ids to current provider tiers and make them config-driven; keep
   `disabled` as a valid provider but record `enrichment_status` in package QA notes so
   absence is visible to ops rather than silent.

## Acceptance criteria

- [ ] Malformed fixture responses on SDK/HTTP paths are rejected by schema validation (test with a missing-required-field payload).
- [ ] Out-of-vocabulary enrichment labels never appear in exported package labels.
- [ ] Curated clips in a package carry captions with provenance + word-count within bounds, or an explicit `caption_missing` QA flag.
- [ ] Enrichment/caption provider and model are recorded per artifact.
