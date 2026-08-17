# Postshot Windows trainer: launch configuration

Everything the Windows lane needs, and where each value comes from. Secrets are
named by their file, never by value.

## Installer, staged 2026-08-17

The Jawset download is a public URL; the Postshot *licence* is a separate login
credential. Only the licence is secret.

| | |
| --- | --- |
| Source | `https://www.jawset.com/public_download/jawset.postshot/win/` (no auth) |
| Bootstrapper | `Postshot-1.1.0.exe`, 46,722,704 B, `sha256:99fc687cf5753f41dce4d6f7b201d9c74893aa25ab2a73fdc3ec85796d1ac38a` |
| Format | InstallShield self-extracting archive |

The bootstrapper cannot be used directly: it hangs in quiet mode, which is why
commit #287 carved the MSI out. Reproduce the carve with:

- outer InstallShield CAB stub ends at byte offset `1131504`
- carve from that offset to EOF as the payload CAB
- extract its single member

| | |
| --- | --- |
| Carved installer | `Postshot-1.1.0.msi`, 45,768,704 B |
| Digest | `sha256:70d4c35de6ff1296a8c0b4b2d87e84b35579baaf08ae173fa843601a2ea0e361` |
| Authoring | WiX Toolset 6.0.0.0, template `x64;1033` |
| Subject / author | Jawset Postshot / Jawset Visual Computing |
| Staged object | `s3://blueprint/postshot/Postshot-1.1.0.msi` (private) |
| Readback | full-byte re-download re-hashed; matches |

Signed GET URL: `~/.blueprint-secrets/postshot_installer_get_url` (0600, 7-day
expiry from 2026-08-17). Digest also at
`~/.blueprint-secrets/postshot_installer_sha256`. The URL carries an
`X-Amz-Signature` and must never enter a tracked file.

## Instance

| | |
| --- | --- |
| Region | `us-east-1` |
| Instance types | `g6.xlarge` (L4 24 GB) preferred, `g5.xlarge` fallback |
| Base AMI | `ami-0ed0165f19a049904` — Windows_Server-2022-English-Full-Base-2026.07.15, verified available 2026-08-17 |
| Quota | G/VT on-demand 8 vCPU verified; `g6.xlarge` is 4 vCPU. Spot quota is 0, so on-demand only |
| Account | `111710313013`, identity `user/Agent` |

Networking uses the account's default VPC (`vpc-01ebd4d2958bb0d23`, 6 subnets,
3 security groups). Pick one subnet and one security group explicitly; the
provider fails closed rather than choosing.

No IAM instance profile is required. The trainer talks only to signed URLs and
makes no AWS API call, so a role would be standing credentials it never uses.

## Environment

```bash
export BLUEPRINT_AWS_REGION=us-east-1
export BLUEPRINT_AWS_ACCOUNT_ID=111710313013
export BLUEPRINT_AWS_INSTANCE_TYPE=g6.xlarge
export BLUEPRINT_AWS_AMI_ID=ami-0ed0165f19a049904
export BLUEPRINT_AWS_SUBNET_ID=<chosen-subnet>
export BLUEPRINT_AWS_SECURITY_GROUP_IDS=<chosen-sg>
export BLUEPRINT_AWS_WORKER_PLATFORM=windows
export BLUEPRINT_AWS_HOURLY_RATE_USD=<verified-current-rate>
export BLUEPRINT_AWS_MAX_HOURLY_RATE_USD=<authorized-ceiling>
```

Bootstrap inputs, read from the worker spec env:

```bash
BLUEPRINT_WINDOWS_NVIDIA_DRIVER_GET_URL=<signed url to the datacenter driver>
BLUEPRINT_WINDOWS_POSTSHOT_INSTALLER_GET_URL=<contents of postshot_installer_get_url>
BLUEPRINT_WINDOWS_POSTSHOT_INSTALLER_SHA256=70d4c35de6ff1296a8c0b4b2d87e84b35579baaf08ae173fa843601a2ea0e361
BLUEPRINT_WORKER_HARD_TTL_SECONDS=<ttl>
BLUEPRINT_POSTSHOT_LICENCE_GET_URL=<signed url to a one-shot licence object>
```

The licence object holds only `POSTSHOT_LOGIN_EMAIL` and
`POSTSHOT_LOGIN_PASSWORD` from `~/.blueprint-secrets/postshot.env`. The worker
deletes it on acknowledgement, and the bootstrap refuses any secret-shaped key
in the spec env because EC2 UserData is readable over IMDS and via
`DescribeInstanceAttribute`.

The NVIDIA datacenter driver still needs staging. Candidate URLs are recorded in
`scripts/postshot_windows_worker/launch_postshot_worker.py`; mirror one to the
same bucket and pin its digest the way the installer is pinned.

## Spend controls

Authorized ceiling: **$50**, 2026-08-17. Bind it into the allocator admission;
do not rely on it being remembered.

Per run: `retry_cap=0`, hard TTL, independent watchdog armed, provider-zero
verified before allocation and again after teardown, teardown unconditional.
An inventory call that fails or returns an unparseable shape counts as unknown,
never as zero.

Expect roughly $1/hr. Provisioning consumes 30–45 minutes of the paid window
because the host is built at boot; a baked AMI removes that once one exists.

## Licence expiry

Postshot Studio, 1 instance, expires **2026-09-01** and is already cancelled.
Shared Access is enabled for `ohstnhunt@gmail.com`, which is what lets the
licence authenticate from a cloud instance. Any paid run must happen before
that date.
