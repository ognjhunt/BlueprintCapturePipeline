# WAM Episode Consistency Scorer

## Purpose

Forward/inverse episode consistency is scored outside WAM execution and outside
the OSCAR/Cosmos WAM evaluator. The evaluator prepares an episode-consistency
request, an external VLM or human-review command scores the generated episode,
and the evaluator only normalizes that command output into proof-bound artifacts.

This keeps three claims separate:

- WAM execution produced or replayed a generated rollout video.
- A success judge labeled whether the generated video appears to complete the
  task.
- An external episode-consistency scorer judged whether the generated video is
  forward/inverse consistent with the action/trace context.

None of these are raw capture evidence, physical robot readiness, deployment
readiness, safety approval, SRCC proof, or real-world validation.

## Artifacts

The OSCAR/Cosmos WAM evaluator writes:

- `wam_episode_consistency_request.json`: scorer input contract containing
  generated rollout video references, task prompts, source trace paths, trace
  summary, and claim boundaries.
- `wam_episode_consistency.command.json`: external scorer output when a scorer
  command is configured.
- `wam_consistency_checks.json`: evaluator-normalized consistency result.

The evaluator must not mark `forward_inverse_consistency_proven=true` unless an
external scorer command ran and returned passing rollout checks with both visual
and action-trace evidence.

## External Command Contract

Configure the evaluator with a separate scorer command:

```bash
BLUEPRINT_ALLOW_WAM_EPISODE_CONSISTENCY_SCORING=true \
python -m blueprint_pipeline.oscar_cosmos_wam_evaluator \
  --input-job-dir <mujoco_or_provider_job_dir> \
  --allow-wam-model-run \
  --wam-consistency-command "blueprint-label-wam-episode-consistency-gemini" \
  --allow-wam-consistency-scoring
```

The evaluator passes these environment variables to the scorer command:

- `BLUEPRINT_WAM_CONSISTENCY_INPUT`: path to
  `wam_episode_consistency_request.json`
- `BLUEPRINT_WAM_CONSISTENCY_OUTPUT`: expected scorer output path
- `BLUEPRINT_WAM_CONSISTENCY_JOB_DIR`: evaluator output directory

The scorer command should write JSON with this shape:

```json
{
  "schema_version": "wam_episode_consistency.command.v1",
  "status": "completed",
  "provider": "external-vlm-episode-consistency",
  "model": "provider-model-name",
  "rollout_checks": [
    {
      "rollout_id": "rollout_1",
      "forward_consistent": true,
      "inverse_consistent": true,
      "confidence": 0.9,
      "rationale": "Visible motion follows the provided action trace.",
      "visual_evidence_used": true,
      "action_trace_evidence_used": true
    }
  ]
}
```

## Gemini Scorer

The repo includes a Gemini-backed scorer command:

```bash
BLUEPRINT_ALLOW_GEMINI_WAM_EPISODE_CONSISTENCY=true \
blueprint-label-wam-episode-consistency-gemini \
  --input <wam_episode_consistency_request.json> \
  --output <wam_episode_consistency.command.json>
```

It reads the API key from `GEMINI_API_KEY`, `GOOGLE_GENAI_API_KEY`,
`GOOGLE_AI_API_KEY`, or the matching `*_FILE` env vars. It does not write raw
credentials or secret hashes into artifacts.

## Boundaries

- The provider runtime does not score forward/inverse consistency.
- The evaluator does not score forward/inverse consistency by itself.
- Generated-video success labels do not prove forward/inverse consistency.
- Visual smoke only decides whether a generated rollout is reviewable enough to
  send to the external scorer.
- A passing consistency label is an episode-level support label, not task
  success, physical robot readiness, deployment readiness, SRCC proof, or safety
  approval.
