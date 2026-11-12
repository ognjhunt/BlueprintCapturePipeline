# Blueprint WebApp Operations

Use when work in this repository touches the contract with the WebApp: the job's intake, the job page, the site's permissions, or what the Pipeline sends back.

Primary references:
- The shared block of `Blueprint-WebApp/PLATFORM_CONTEXT.md`
- `Blueprint-WebApp/AGENTS.md`
- `BlueprintCapturePipeline/docs/SYSTEM_MAP.md`
- `BlueprintCapturePipeline/docs/webapp_handoff/`

Required behavior:
- Treat the WebApp as the site side: the job page, the first look on paper, the site's permissions and the responses of deployment companies.
- Keep what the site sees in line with what the Pipeline's records show. Tell the site plainly where the job stands.
- Send a site's video or simulation to a deployment company only with the site's permission, and only to companies Blueprint has approved.

Do not:
- Present the simulation or the early trial as the site's answer.
- State a price, or a response that has not been given.
- Assume static page text is the same as the job's live state.

Output:
- WebApp-facing notes or changes that match the shared block.
