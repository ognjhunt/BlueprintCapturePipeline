# SAM contribution source disclosure

ADP-009D contribution measurement precedes source-object cutout. Its provider
bundle contains the complete source scene converted to standard PLY. Conversion
preserves every Gaussian; permission to disclose derived frames does not cover
this payload. General compute/spend authorization does not establish data rights.

The contribution worker now fails with
`sam31_contribution_disclosure_explicit_full_source_authority_required` before
creating its bundle or reaching staging/allocation unless the task's
`human_authority.full_source_provider_disclosure_authority` references a separate,
retained `public_scene_full_source_provider_disclosure_authority.v1` record.
Source selection, local conversion, calibrated views and separately admitted SAM
frame tracking remain independent earlier phases. Historical terminal receipt
readback remains available.

An admitted successor record must bind exact original and converted hashes/sizes,
the complete Gaussian count, publisher scene/revision, execution commit, Vast,
and the contribution purpose. It must explicitly authorize full source-content
private processing and identify a retained publisher-rights basis, with actual
publisher-terms and private-processing-permission evidence files bound by hashes
and sizes. Publisher permission cannot be manufactured from a user's willingness
to pay. Public redistribution and provider training remain forbidden. The
validator reopens the evidence files and requires the terms hash to match the
conversion's retained terms.

The original conversion receipt stays unchanged, including its local-only scope
and `raw_private_upload_authorized=false`. A later separately authorized
processing scope is retained as a successor; it never rewrites historical
conversion truth. The worker binds that readback into its execution authority,
checks existing bundle/source/authority identity before staging, and describes the
payload as `full_source_scene_reencoded_standard_splat`.

The current Scene 841757 task has no such full-source permission. This guard
therefore blocks its remote contribution phase. Nothing in this document grants
permission or directs an operator to fabricate the required authority.
