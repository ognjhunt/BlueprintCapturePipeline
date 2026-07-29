# Decision/Evidence Router WebApp handoff v1

Pipeline owns claim decomposition, testbed/version validation, method
qualification, evidence routing, leaf EvaluationRunSpec compilation, result
normalization, disagreement handling, scientific verdicts, claim ceilings, and
the final Decision Envelope.

WebApp may:

- collect the decision question, claims, candidates, limits, rights/privacy
  restrictions, and permitted evidence-method families;
- submit an exact `decision_evidence_request.v1`;
- display the persisted plan/run state and final envelope;
- render accepted versus rejected evidence, disagreements, uncertainty,
  abstentions, next experiments, and physical-evidence requests.

WebApp must not select providers, infer qualification from availability,
recompute scientific verdicts, merge correlated evidence as independent votes,
or raise a claim ceiling.

Files:

- `request.schema.json`: public request contract.
- `normalized-evidence-result.schema.json`: stable method-result envelope.
- `result.schema.json`: public Decision Envelope contract.
- `status-state-machine.json`: allowed public states and transitions.
- `legacy-field-translations.json`: fail-closed legacy input migration.
- `security-redaction-rules.json`: fields and behaviors that may never cross the
  handoff.
- `compatibility-policy.json`: versioning and deprecation policy.
- `examples.json`: decision, abstention, and partial-decision examples.
- `artifact-manifest.json`: SHA-256 inventory for this handoff version.

The maintained Site-Task Testbed is a versioned substrate referenced by the
request. It is not a separately marketed product. Post-training is an allowed
evidence use only when the Decision Envelope says the rights, provenance,
alignment, quality, and leakage gates pass; no handoff artifact implies that
training occurred or that a policy improved.
