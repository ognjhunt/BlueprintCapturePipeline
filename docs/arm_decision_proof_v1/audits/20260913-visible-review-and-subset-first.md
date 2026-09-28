# Visible object guidance and subset-first training admission

Scope: ADP-009B public-scene appearance-removal prerequisite for the day-21 transition. The completion artifact remains an independently reviewed object-free 3D scene; transport tests and subset coverage are not final appearance qualification.

## Saved production evidence

On attempt source-5cbbcbff25476c3132d4fe23, GPT-image-2.5 received original images containing both the small dark vase and the taller white vase in source-04, source-07, source-11 and source-12. The exact alpha masks identify only the dark vase. The returned full images remove both vases. The independent pretraining reviewer incorrectly accepted all 16 frames. The user and operator rejected those four, and GPU50946765 was terminated before proceeding with those inputs. Historical model verdicts are preserved rather than rewritten.

The vision review payload sent the API mask PNG unchanged: RGB is uniformly white, while the target exists only in alpha. That representation can become blank if alpha is dropped or composed onto white. It also omitted the target's scene-derived description. Review now receives a visible white-target/black-background mask and the same target description, with source and candidate images unchanged. The original editor mask remains untouched and digest-bound. Supported mask encodings are handled explicitly.

The editor's new versioned prompt emphasizes removing one named object and retaining separate objects overlapping or hidden behind it, including reconstructing their revealed portions. Historical prompt versions retain their original text. Repair requests include the original target instruction.

## Use sufficient good views

The existing requirement is at least 8 and 60 percent approved views, plus two approved camera axes within 45 degrees of each excluded view. Twelve of sixteen pass the count requirement. On the recorded cameras, excluding 04/07/11/12 leaves only source-07 undercovered. Repairing source-07 can provide 13 approved targets while excluding the other three bad edits; that remains conditional on review of the repaired frame.

Training admission now tries the existing coverage-checked subset before paying for repairs. If coverage or count is insufficient, the repair planner selects uncovered views and any additional views needed for the count floor. It never fabricates accepted reviewer rows. The repair request records the selected and deferred failures accurately. All cameras remain in the final review; excluded edited frames cannot become training teachers. Existing masked original-anchor behavior and final visual review remain unchanged.

## Verification

Focused tests cover visible mask transport for alpha, white-on-black and black-on-white encodings, unchanged source/candidate image bytes, named target transport, prompt compatibility, valid-subset admission without repair calls, bounded recovery, one-view coverage repair planning, and selecting a non-prefix repair subset with correct deferred-camera bookkeeping. Saved-camera replay identifies only source-07 as requiring repair. A live review replay is still needed to establish that the updated reviewer catches the observed model failures; code tests alone do not prove model judgment.
