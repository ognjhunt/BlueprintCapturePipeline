# Company Policy Container Contract v2

This handoff freezes the smallest company-facing policy interface for the
Arm Decision Proof partner adapter (ADP-011 and ADP-050). It is not launch
authority and it does not make Blueprint a general container hosting platform.

The company supplies an OCI image reference pinned by SHA-256 plus declared
robot, camera, state, action, timing, rights, and resource semantics. The image
contains the company's policy code and weights. Blueprint supplies observations
through a versioned local proxy and owns action validation, simulator execution,
evidence, scoring, watchdogs, and teardown.

## Deliberate separation

The immutable contract contains no registry credential, environment secret,
host path, mount, external endpoint, or launch request. A private image uses a
separate short-lived, read-only, single-use registry credential lease. The lease
is referenced by opaque identifier only after both artifacts are admitted.

A valid contract means only `development_only contract admitted`. Before any
real scene observation, separate receipts must prove:

1. rights and candidate identity;
2. credential-lease validity without exposing credential bytes;
3. image digest readback;
4. sandbox hardening and measured no-egress behavior;
5. synthetic protocol conformance;
6. launch profile and paid-attempt authority;
7. terminal evidence, teardown, image removal, and provider zero.

## Fixed wire protocol

The initial live surface is intentionally one protocol:

- container listens on its own loopback port;
- Blueprint reaches it only through the Blueprint-owned proxy;
- action requests use `POST /v1/actions` under `http_json_v1` protocol `1.0`;
- observations contain only declared RGB, state, and prompt fields;
- raw scene assets, capture roots, evidence roots, Docker socket, and
  credentials are never mounted into the container;
- every camera is lossless RGB with calibration identity;
- action rows, channel order, units, raw accepted bounds, executed bounds, and
  normalization semantics are explicit.

New transports or action types require a versioned adapter and tests. They are
not caller-selected strings.

## WebApp implementation contract

The WebApp should render `contract.schema.json` as a guided form and prefill
known robot templates, while keeping an advanced JSON editor for exact review.
The UI must submit candidate metadata separately from a private-registry
credential. It may display the normalized contract and digest, but must never
echo the credential or imply that contract admission means launch readiness.

The corresponding Pipeline intake will accept the normalized contract and an
opaque credential-lease ID only after tenant ownership, idempotency, expiry,
single-use, rights, and digest checks pass. GPU allocation remains a separate
governed action.
