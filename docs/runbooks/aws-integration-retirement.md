# AWS account integration retirement

Blueprint AWS account access is retired. Cost Explorer collection was removed
in PR2519; the remaining EC2 provider, inventory queries, Postshot execution,
watchdog, standalone worker, and AWS credential/runtime requirements are removed
or retained as locally refusing compatibility entrypoints. New AWS/Postshot
requests fail before detachment, provider probes, staging, or paid admission.
Completed Postshot publication evidence can still be adopted with its original
digest validation. No resource, capture, secret, or retained receipt is deleted.

S3-compatible transport remains supported for explicit HTTPS B2, Spaces, R2,
and RunPod endpoints with explicit credentials. Clients refuse the default AWS
endpoint and known AWS endpoint domains before SDK construction; they never use
ambient SDK credential discovery. Standard S3 SDK parameter names are preserved.
Public vendor downloads hosted on AWS remain vendor-owned transport, without
Blueprint AWS account credentials or billing access.

AWS inventory is unavailable and its resource count is unknown. A provider-zero
receipt for active providers does not prove AWS resource absence. Historical AWS
billing and allocation evidence remains historical; it cannot authorize spend.
AWS account closure and any resource/data destruction require a separate owner
inventory and preservation decision.

Release evidence archival also refuses AWS. Its workflow requires an explicit
compatible endpoint and private credential files; it fails closed if the owner
has not configured an existing compatible Object Lock destination. This change
does not create credentials, a bucket, a retention policy, or replacement spend,
and does not claim that a compatible archive is currently configured. Prior
immutable retention receipts remain unchanged.
