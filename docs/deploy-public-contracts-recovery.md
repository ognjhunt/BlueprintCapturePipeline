# Recover the reviewed usage fix deployment

ADP-081, partner-proof day-42 prerequisite: release the reviewed immutable usage
replay repair (#2597). This does not complete the economics gate.

Two guarded deployments of merged b0ce12c74251f63db5f50ff55100597814656c9b
refused with deploy_scene_retirement_runtime_unproven. Authorized reads show the
protected source snapshot and wheel inputs exist, while the pinned public
BlueprintContracts bare repository has an empty FETCH_HEAD and no objects.
The installer hard-codes SSH acquisition. The exact pin is publicly reachable
over HTTPS. The precise host SSH failure is not exposed by the bounded wrapper.

Contract before editing:

- Trigger: the existing reviewed-main deployment, before active-link activation.
- Input: the same locked Contracts repository/commit and authenticated installer
  blob from the selected release. No credential, scope or cleanup-authority change.
- Acquire the public dependency through fixed HTTPS without user configuration,
  credential helpers, prompts or hooks. Continue verifying every Git object hash.
- A valid retained installer must not trap an upgrade in old acquisition code.
  Authenticate the selected release installer into a protected immutable per-commit
  directory; validate existing retained installer state before selecting it.
  Execute the authenticated candidate via its retained descriptor. Preserve the
  existing prepare/refresh and root installer publication protocol.
- Existing unknown, symlinked, writable, changed or unbound installer state still
  refuses. A failed candidate leaves the existing installation available; no
  success receipt or active-link movement follows a failed preparation.
- Retrying the same candidate reuses exact authenticated bytes. Modified candidate
  bytes conflict. Existing bounded subprocess/deployment deadlines remain.
  The existing hard-link publication window remains fail-closed: a crash between
  link creation and temporary-name removal can retain two links and require an
  owner to reconcile that exact protected partial state. This patch does not
  claim automatic recovery of every filesystem crash window.
- Completion requires the existing source-bound prepared/refreshed receipt with
  authority_issued=false and cleanup_enabled=false, then all normal release checks.
- Tests cover a real hermetic Git acquisition, first installation, valid old helper
  upgrade, mutable checkout exclusion, unknown retained helper refusal and changed
  candidate refusal. Run impact selection and its mandatory Linux proof lane.
- Independent review and passing required CI precede merge/deploy. Preserve explicit
  dispatch stops and configured-controls pause. Rollback preserves immutable inputs
  and prior installation; no deletion, paid canary or first-contact activation.

Ownership: this isolated deployment-repair branch is a dependency of #2597. Fresh
open-PR and source-history inspection found no competing installer-fetch change.
Existing broad CI repairs and research work remain with their existing owners.
