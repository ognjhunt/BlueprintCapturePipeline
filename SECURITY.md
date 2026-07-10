# Security Policy

## Supported versions

Security fixes are provided for the current `2.x` release line. Older release
lines and unversioned snapshots are unsupported.

| Version | Supported |
| --- | --- |
| `2.x` | Yes |
| `< 2.0` | No |

## Reporting a vulnerability

Do not open a public issue or pull request containing vulnerability details,
credentials, private capture data, or a working exploit.

Use the repository's
[private vulnerability reporting form](https://github.com/ognjhunt/BlueprintCapturePipeline/security/advisories/new).
Include the affected version or commit, the reachable component, reproduction
steps, impact, and any suggested mitigation. If the private form is not
available, contact the repository owner through GitHub and request a private
reporting channel without disclosing the vulnerability in public.

The maintainers will acknowledge a complete report, assess severity and
affected versions, coordinate a fix and disclosure plan, and credit the
reporter when requested and appropriate. Please allow time for a patch to be
prepared before publishing details.

## Scope

Reports about dependency vulnerabilities are useful when they identify a
reachable Blueprint code path or a deployment-specific impact. Do not include
real customer captures, secrets, access tokens, or personal data in a report;
use minimal synthetic reproduction data.
