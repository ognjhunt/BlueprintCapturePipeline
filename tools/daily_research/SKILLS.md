# Reviewed mounted skills

The saved template uses the documented Agents API file-discovery path:
`capability_directories=["/workspace/capabilities/blueprint"]` and four inline
files. Its empty `skills` and `plugins` lists are expected, not evidence of lost
skills. Do not register new versions or mutate the template to fix preflight.

The four files packaged under `capabilities/` are the exact instruction-only
files approved September 30, 2026. All recovered bytes match the mounted-file
hashes in the [preserved validation receipt](https://app.notion.com/p/3eb80154161d81b4a0ddff9bdcfe5af6).
The Blueprint files and license were recovered from preserved inline review
text; deep-research was fetched from its original immutable source commit
`42dd24080fce6d731d00e2a1134f398c3da4171b`. Its MIT notice is preserved.
No executables, hooks, packages or credentials are included.

| Path relative to the capability directory | Bytes | SHA-256 |
| --- | ---: | --- |
| blueprint-evidence-qualification/SKILL.md | 3965 | 5fdc5f3c6dde9cc252cd43bd4165d09b4b2d5686835b70961d00e0d7e91f0756 |
| blueprint-evidence-qualification/references/prospect-contract.md | 1838 | 34cf345832dea8226d14ed31b3197a44c293ac91dc903fb596a9b89d05299258 |
| deep-research/LICENSE | 1072 | 3a9cf254e155282014880e9569b9039bc17ce6a43919df23741cf14d24481244 |
| deep-research/SKILL.md | 5386 | 2646cdf3942d918e84febf020b289fbfb7b5cf601e43ee7e7349e6c5105941c5 |

Preflight performs only the two existing provider GETs. It verifies the exact
template ID, disabled network, capability directory, inline paths/sizes and
absence of separately attached skills/plugins, then hashes packaged bytes.
Template GET does not expose inline contents: `template_inline_content_verified`
remains false. A metadata match alone is never reported as current remote byte
verification. The historical validation did read these hashes; it did not prove
automatic skill selection or the actual small runtime tier.

Each future authorized create request includes the four hash-verified inline
files as documented session overrides, with the same directory and disabled
network. The existing durable intent includes the entire request before its one
provider-create attempt. This prevents reliance on opaque template file contents
and preserves exact reviewed versions without a provider template mutation.
Actual hosted mounting/skill loading still needs evidence from the authorized
canary's exact root turn. No canary has run for this correction.

The original validation session
`sess_08854dcb5c638b91006abc8014644c81958bca9d509f75de3c` has a preserved,
action-time-approved deletion receipt (HTTP 200, then session/environment GET
404 on September 30 at 03:37:57 UTC). That receipt clears only that exact session;
it cannot clear the currently observed unclassified saved-agent session.

Sources: [Agents file-based skills](https://developers.openai.com/api/docs/guides/tools-skills#agents-api),
[template/session overrides and inline files](https://developers.openai.com/api/reference/resources/beta/subresources/agents/subresources/sessions/methods/create),
[original skill review](https://app.notion.com/p/3eb80154161d81eea84bf22f330dc458).
