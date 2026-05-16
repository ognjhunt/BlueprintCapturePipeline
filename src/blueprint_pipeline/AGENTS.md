# Agent Guide For `src/blueprint_pipeline/`

This package owns materialization, package assembly, provider adapters, privacy
and geometry lanes, runtime support, launch gates, and WebApp sync.

Keep raw capture truth, canonical package truth, provider-preview projection,
hosted-review projection, and gate snapshots separate. Compatibility fields with
`qualification` names may remain, but they are support artifacts rather than the
product center.

Do not hardwire Blueprint to one provider, checkpoint, GPU path, or hosted
service. Do not promote fallback geometry, raw provider bypass, mocked Stripe,
missing SDK env, or automated contract checks into live launch proof.
