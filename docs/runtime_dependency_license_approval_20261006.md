# Runtime dependency license approval, 2026-10-06

The policy owner, @ognjhunt, approved deployment and merge at 10:12 UTC on
2026-10-06 after the explicit license review request at 23:50 UTC on
2026-10-05 and the release decision summary at 10:11 UTC on 2026-10-06.
The reply was “Deploy and merge.” The preceding request identified all five
exact versions below and described the custom SAM license obligations.

| Exact runtime component | Reviewed license |
| --- | --- |
| AnyIO 4.14.2 | MIT |
| google-auth 2.58.0 | Apache-2.0 |
| PyJWT 2.15.0 | MIT |
| urllib3 2.8.0 | MIT |
| meta-sam-parser 0.0.5 | SAM License, last updated November 19, 2025 |

The SAM license is recorded as `LicenseRef-Meta-SAM-2025-11-19`, rather
than an SPDX license for a different set of terms. Its reviewed source is
the [official SAM license](https://github.com/facebookresearch/sam3/blob/main/LICENSE).
The repository also retains its license text at
`src/blueprint_pipeline/_sam_parser_js/LICENSE`.

The reviewed SAM terms include providing the agreement when distributing
SAM materials, research attribution, applicable privacy and trade controls,
use restrictions, no warranty, and indemnity. This record reflects the
user's approval for the Blueprint capture and world-model workflow; it is
not a claim of legal counsel review or independently retrieved package
metadata. Existing notice and rights obligations still apply.

The five exact-version records use the approval date and the policy's
existing one-year review convention, expiring on 2027-10-06. The superseded
AnyIO, google-auth, and PyJWT runtime records are removed from the active
policy and remain in Git history. The existing urllib3 2.7.0 review remains
because the reconstruction worker lock still pins that exact version.
All other records and fail-closed checks remain unchanged.

This approval covers the described normal merge and release. It does not
authorize new spending, model or paid processing runs, new credentials or
security access, manual suspension or cancellation, or changes to protected
run windows and the founder dispatch hold.
