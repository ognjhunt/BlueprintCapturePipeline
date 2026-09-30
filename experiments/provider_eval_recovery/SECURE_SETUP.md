# Exact current setup checkpoint

Observed in the current running Pipeline process on 2026-09-30, without reading
values. These booleans do not describe the user's vault inventory or establish
that a previously saved secret is missing. The published environment version and
vault request metadata are not exposed by the available executor tools.

| Binding | Present |
| --- | --- |
| `PARALLEL_API_KEY` | false |
| `PERPLEXITY_API_KEY` | false |
| `OPENAI_API_KEY` | false |
| `OPENAI_PROJECT_ID` / `OPENAI_PROJECT` | false |
| Existing Default project environment binding matches required ID | false |
| `HTTP_PROXY` / `HTTPS_PROXY` | true |

The parent supplied the supported private setup route:
**Settings > Codex Cloud > Personal vault > Add > Network secret**. Reuse the
existing provider keys, restrict each to the matching API hostname below, and
set Applies To to **BlueprintCapturePipeline only**. A fresh configured executor
should check only binding presence after the user reports saving; this running
session may not receive updated bindings. No keys have been saved/confirmed
through this recovery. A successful harmless recovery check after the reported
disconnects found the branch/files intact and shell responding.

The required OpenAI project is **Default**,
`proj_F2tFJuxLaovJru8RrtXRaqNj`; model is **`gpt-6.1-sol`**.
The parent later reported the user's authorization for one dedicated Default
OpenAI key; the user created it himself. Existing keys remain untouched, and this
implementation task performs no duplicate key setup. Parallel/Perplexity reuse
existing accounts. No IAM changes, new grants, subscription or top-up are authorized.

| Provider | Allowed host and endpoint | Required request headers | Required existing access |
| --- | --- | --- | --- |
| Parallel | `api.parallel.ai`, `POST /v1/search` | `x-api-key`; `Content-Type: application/json` | Existing Search API key/account; Fast and Advanced |
| Perplexity | `api.perplexity.ai`, `POST /search` | `Authorization: Bearer`; JSON content type | Existing Search API organization/key; `search_type: fast` and `web` |
| OpenAI | `api.openai.com`, `POST /v1/responses` | `Authorization: Bearer`; `OpenAI-Project: proj_F2tFJuxLaovJru8RrtXRaqNj`; JSON content type | Existing Default project key with Responses/model use for `gpt-6.1-sol`, Standard tier |

No key-management, payment, subscription, Task, Agent API, private search, CRM,
connector or deployment endpoint is needed for this raw comparison. Existing
permissions must already allow these operations; do not broaden them here.

Two transport-supported injection paths are implemented and mock-tested:

1. Owner-configured existing process/environment secret bindings named
   `PARALLEL_API_KEY`, `PERPLEXITY_API_KEY`, `OPENAI_API_KEY`.
2. Owner-mounted existing private secret files, referenced by
   `PARALLEL_API_KEY_FILE`, `PERPLEXITY_API_KEY_FILE`, `OPENAI_API_KEY_FILE`.
   Each must be a regular nonsymlink file, owned by the executor user, with
   no group/other access (normally mode 0600). File references are configuration,
   not secret values. The code never creates or writes the key files.

Use exactly one of the environment/file routes per provider. Secret values must
be entered through the environment owner's existing secure configuration or
mounted secret store, never pasted in chat, shell command text, task prompts,
Git, logs, receipts or PRs. This task has not configured either route.

**Network-secret proxy capability is unverified.** This executor exposes an
HTTPS proxy, but no Network-secret configuration/readback tool or documented
header-injection contract is available here. Do not assume Bearer-only injection
supports Parallel. If the owner's existing proxy UI supports per-host custom
headers, configure the exact host/header mapping above outside this task and
retain a sanitized capability receipt. Otherwise the mock-tested private-file
route supplies `x-api-key` in process memory while normal HTTPS still uses the
configured proxy; owner approval/configuration of that mount remains required.
No proxy denial is bypassed, and no direct-network override is implemented.

The exposed OpenAI connector offers new-key setup/creation, not import of these
existing keys; those tools were not called. The user handled the one authorized
OpenAI creation himself. No
turnkey existing-key installation tool is exposed for this selected executor.

Live dispatch remains blocked until an existing-access receipt confirms the
exact Default project, these three hosts and Parallel custom-header support
(whether injected by proxy or supplied by the mounted-file transport), and a
pinned GPT6.1Sol input-token counter is supplied. It also requires the shared
paid-resource admission grant bound to the immutable public-input/rubric plan.
The Python HTTP seam is implemented; a live end-to-end comparison command is
not yet wired to a pinned tokenizer and parent-side isolated inference rubric.
The actual public-case schema is integrated. No live calls or credential loading
have occurred in this implementation task.

The comparison needs OpenAI **Responses Write** (`POST /v1/responses`);
Models Read is not used. Existing saved-agent session/turn permissions belong to
the separately maintained research runner and must be matched to its own calls.

Budget is already approved: **$10 cumulative across pilot plus remaining18**,
including all three providers. Pilot cap is **$1**, exactly eight raw provider
attempts, no pilot retry. Same journal/scope retains accepted or ambiguous spend;
pilot cells must be adopted and never rerun for the remaining18. Full conservative
subtotal is $8.9952 including cache-write and provider retry bounds; an extra
$0.50 provider allowance gives $9.4952, leaving $0.5048 for tax/FX. Pilot subtotal
with the per-attempt extra allowance is $0.90052, leaving $0.09948 within $1.
No auto top-up or subscription is authorized. Any newly discovered fee must fit
the remaining approved cumulative cap or stop execution.

Library ZIP bytes are still blocked: the supported Library helper returned
`library file transfer failed: download failed`. Verbatim public JSON supplied
in the delegation can resolve portability without private-URL guessing. Its own
byte/hash identity must be frozen separately, preserving the supplied ZIP hash
as **unverified original-bundle provenance**, not claiming recreated ZIP parity.
The parent subsequently provided the exact provider-safe public JSON. It is
retained in `real_public/inputs.parent-message.json`; its actual SHA256 exactly
matches the original declared public-file SHA256
`cac8d7a31aea2ad1c2e5a47ea37e434abcaae4b4910ba4afc8f81dd11c404ff3`.
All20 real public cases and80 request envelopes are prepared. This is a
parent-message route, not successful Library ZIP materialization. The actual
oracle stays parent-side; the live controller does not receive it.
