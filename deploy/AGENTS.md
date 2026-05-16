# Agent Guide For `deploy/`

`deploy/` contains live-risk infrastructure for privacy and geometry runner
services. Treat Docker, Terraform, Cloud Run, and deployment scripts as
production-adjacent unless the user says otherwise.

Do not edit secrets, provider tokens, `.env*`, GPU runner credentials, Terraform
state, or live deployment config without explicit approval. Prefer documenting
required env and validation evidence over making live changes.

Fallback/local runner success is not live GPU proof. Before claiming live proof,
verify provider-native outputs and the required geometry/privacy readiness labels.
