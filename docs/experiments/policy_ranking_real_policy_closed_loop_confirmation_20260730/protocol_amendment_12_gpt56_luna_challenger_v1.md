# Protocol amendment 12: GPT-5.6 Luna independent challenger

Frozen prospectively on 2026-07-30 before any judge request, complete virtual
episode, pair inventory, prediction, or physical-label join existed.

## User-authorized change

The independent OpenAI challenger changes from the underspecified `full GPT-5`
arm in `protocol_v1.json` to the explicitly named `gpt-5.6-luna` arm. Gemini
remains the primary judge and still runs first.

The registered judge order is now:

1. `gemini-3.6-flash` primary, native generated-only MP4, Gemini Batch API;
2. `gpt-5.6-luna` independent OpenAI challenger, deterministic generated-only
   frames, OpenAI Responses Batch API.

Both arms must receive the same frozen unordered policy-pair inventory, episode
membership, task text, and side randomization. Neither arm receives policy
identity, physical outcomes, or physical ground-truth pixels.

## Verified public capability boundary

Official provider documentation inspected on 2026-07-30 reports:

- `gemini-3.6-flash` is a stable model with text, image, video, audio, and PDF
  input; structured output, thinking, and Batch API support;
- `gpt-5.6-luna` supports Responses, Batch, image input, structured outputs,
  and reasoning tokens, but not video input;
- OpenAI standard token pricing for `gpt-5.6-luna` is USD 1.00 per million input
  tokens, USD 0.10 per million cached input tokens, and USD 6.00 per million
  output tokens; the Batch API advertises a 50 percent discount;
- Gemini 3.6 Flash standard pricing is USD 1.50 per million input tokens and USD
  7.50 per million output tokens; Gemini Batch advertises 50 percent of standard
  cost.

Authoritative URLs:

- https://developers.openai.com/api/docs/models/gpt-5.6-luna
- https://platform.openai.com/docs/api-reference/batch/object?api-mode=responses
- https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash
- https://ai.google.dev/gemini-api/docs/batch-api

Provider availability, exact account access, current pricing, rate limits,
retention/data treatment, and any dated snapshot or resolved backend identity
must be fetched and frozen again immediately before the transport canaries.
This amendment does not treat documentation inspection as account or billing
validation.

## Frozen transport

Gemini receives each episode's native generated-only MP4 when its transport
canary verifies ingestion. Luna receives 32 deterministic generated-only frames
from episode A and 32 from episode B, for 64 image inputs per pair. Luna receives
no video because its official model contract does not support video input.

The existing concise structured comparison schema, sufficient output-token
allowance, bounded one-repair rule for token-exhausted rows, redaction,
idempotency, retry accounting, and seven-pair label-blind Batch pilot remain
mandatory. Exact reasoning effort and image detail are frozen before the pilot
and cannot change after results are observed.

## Budget and graph rule

The combined Gemini plus OpenAI evaluator cap remains USD 25. Gemini runs first.
After reconciling its actual cost, Luna's complete matrix plus the frozen repair
reserve must fit the remaining cap. No partial Luna graph earns ranking,
Bradley-Terry, or cross-family credit.

## Claim boundary

OpenAI describes Luna as its cost-sensitive, high-volume tier. This experiment
therefore calls it the **GPT-5.6 Luna independent OpenAI challenger**, not a
full, flagship, Sol-equivalent, or definitive strongest-OpenAI judge. A complete
Gemini-versus-Luna comparison can establish cross-provider agreement for these
two frozen judges. It cannot establish that Gemini is better than GPT-5.6 Sol,
GPT-5.6 Terra, the previously intended full GPT-5 arm, or every OpenAI model.
