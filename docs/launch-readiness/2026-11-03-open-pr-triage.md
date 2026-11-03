# Open PR triage (2026-11-03)

Beta definition of done, item 8: "Launch-critical PRs are merged or explicitly
deferred by the owner." This page covers every PR that was open at about 04:30Z
on 11-03 in BlueprintCapturePipeline, Blueprint-WebApp and BlueprintCapture: 56
PRs. It leaves out the three open PRs of the cloud session that drives
capture-447 (Pipeline #3196, WebApp #1198 and #1175). None of those were touched.

## How each PR was checked

Each PR's head commit was compared with `main` by content, not by title:

1. `git merge-tree` against `main`: does the PR conflict, add nothing, or add
   content?
2. How many of the PR's added lines `main` already has.
3. For each PR that still adds something, the code on `main` was read to see if
   `main` has the same behaviour some other way.

`main` was Pipeline `d7e2735a9`, WebApp `58677f751` and BlueprintCapture
`870ad89d1`.

## Result

No PR in this list is launch-critical. The launch-critical PRs named in the
brief (section 3, H-02) are all settled, except two that belong to the cloud
session:

| PR | State |
|---|---|
| WebApp #1168 | Merged 11-02 14:04Z. |
| WebApp #1142 | Carried as #1178, merged 11-02 14:35Z. |
| WebApp #1103 | Closed 11-02 as superseded. |
| Pipeline #3226 | Merged 11-02 13:55Z. |
| Pipeline #3042 | Its rights part merged as #3262 (11-02 17:18Z, G-02). Close the rest (table below). |
| Pipeline #3051 | On `main` through #3068 (10-29). Close (table below). |
| Pipeline #1980 | Superseded. Close (table below). |
| Pipeline #3196, WebApp #1175 | Open, owned by the cloud session. #3196 merges after capture-447 settles (C-01). |

The brief's owner decisions: WebApp #1143 was closed and replaced by #1195
(merged 11-02 22:52Z). The Luna-to-Haiku PRs (#931, #918, Pipeline #2644) are in
the table.

So item 8 waits only on #3196 and #1175, and on your choice for the PRs below.

## What to close or defer

Recommended: close 33, defer 23. Closing a PR keeps its commits, so the work can
come back.

| PR | What it does | Do | Why |
|---|---|---|---|
| Pipeline #3042 | Autonomy draft: recovery from the handoff dead-letter queue, funding readback, quote refresh, stale-worker fences | Close | Its rights part merged as #3262 on 11-02. The rest is inside #3056. |
| Pipeline #3045 | Autonomy draft: authored task inputs for task families (AUT-12), off by default | Close | Inside #3056: its head commit is part of #3056. |
| Pipeline #3052 | Autonomy draft: native metrology evidence (AUT-06) | Close | Inside #3056. Its first part merged as #3041 on 10-28. |
| Pipeline #3051 | The screening worker writes a status snapshot around each pass | Close | `main` does this since #3068 (10-29), with failure categories. What is left is small: a check for bad revision numbers, and a snapshot when a pass crashes. Also inside #3056. |
| Pipeline #3037 | Adds a second recording to a job's Atlas input | Close | Your 10-23 rule: the one video is the only input. The WebApp half is #1136. |
| Pipeline #2548 | Two CI fixes from 10-02 (typing, a SAST warning) | Close | `main` already types the AWS list in `gpu_spend_guard.py`. The other change only quiets a SAST warning on a query made from two fixed strings. |
| Pipeline #1980 | Clean-plate stage: reuse finished runs, stop on leftover people | Close | `main` rewrote this stage. Website jobs reuse their saved analysis, and people in admitted footage are not a hold. D-16 dropped person screening (G-03). |
| Pipeline #265, #266, #267, #268, #269 | Dependabot: CUDA 13.3.1 in five GPU images (08-01) | Close | #3160 moves the same five Dockerfiles to 13.4.3. |
| Pipeline #56 | Dependabot: Python 3.14 in the MuJoCo image (07-10) | Close | #3159 changes the same Dockerfile again. |
| WebApp #1136 | Lets a site add extra camera angles to a job | Close | Your 10-23 single-video rule. Its base PR #1137 was closed on 10-28. 12 conflicts. |
| WebApp #1085 | Copies the 10-21 doctrine and a test fixture | Close | The 10-28 doctrine replaced that text (#1127, then #1188 on 11-02). The fixture part repeats #1083. |
| WebApp #918 | Luna-to-Haiku model swap, first version | Close | #931 contains it: its head commit is part of #931. |
| WebApp #915 | Docs record of the 10-07 email-branding release (#913) | Close | An old release note that nothing reads. The branding itself is on `main` (#913). |
| BlueprintCapture #93 | Copies the 10-21 doctrine | Close | #94 copied the 10-28 doctrine on 11-03. |
| BlueprintCapture #54 | July pre-beta iOS fixes: resumable upload, crash reports, disk checks | Close | The app was rebuilt for App Clip capture (#76, 09-22). 14 of the 20 files it edits no longer exist, and it has 23 conflicts. If you want resumable uploads or crash reports, start a new PR from `main`. |
| BlueprintCapture #21, #24, #30, #31, #32 | RoomPlan and AR-session fixes (2025-11) | Close | RoomPlan support was removed on 2025-12-08, and the App Clip rebuild (#76) replaced the other files they edit. |
| BlueprintCapture #27, #28 | ReplayKit fallbacks beside RoomPlan (2025-11) | Close | RoomPlan was removed. The recorder on `main` has its own stop timeout. |
| BlueprintCapture #25 | A timeout when the screen recorder does not stop (2025-11) | Close | `main` has this: an 8-second fallback in `Shared/Capture/VideoCaptureManager.swift`. |
| BlueprintCapture #23 | Keeps the capture screen open until the recording ends (2025-11) | Close | The App Clip rebuild (#76) removed the file it edits. |
| BlueprintCapture #13, #14, #15, #16, #18 | Nearby Targets and search screens (2025-10) | Close | Those screens were deleted on 07-21 as unused code. |
| Pipeline #3056 | All Codex autonomy drafts on one branch (AUT-02/06/08/09/10/12/17) | Defer | Not on the beta path. Keep it as the one record of that work. It is 572 commits behind `main` with 32 conflicts, so it needs a rebuild before any merge. |
| Pipeline #2808 | Isaac Lab GA runtime comparison (ADP-009D) | Defer | Research draft: 0 of 40 valid episodes. It contains #2806. |
| Pipeline #2806 | SimReady Central asset intake (ADP-009) | Defer | Merges cleanly, but nothing in the beta uses it. |
| Pipeline #2740 | Full exact-workcell evaluations for the rigid Franka/DROID profile | Defer | The beta stops at screening. 7 conflicts and open review findings. |
| Pipeline #2736 | FLORA video grading for research | Defer | Research only. 6 conflicts. |
| Pipeline #2714 | Agent instructions for site and contact research | Defer | Instructions only, no product change. Conflicts in `AGENTS.md`. |
| Pipeline #2644 | Luna-to-Haiku model swap (Pipeline) | Defer | Your decision in the brief. `main` still runs `gpt-6-luna`, and it works. |
| Pipeline #3161, #3255 | Dependabot: CodeQL 4.37 to 4.39.1 (init, analyze) | Defer | Small, and the checks pass. Merge the two together after the beta. |
| Pipeline #3160 | Dependabot: CUDA 13.4.3 in seven GPU images, including SAM 3 and reconstruction | Defer | A new major CUDA version under seven GPU images. 2 checks fail. |
| Pipeline #3159 | Dependabot: Python 3.15 in four images | Defer | The brief said to leave the Python 3.15 bump unmerged (#3158, closed 11-02). |
| Pipeline #2706, #2709, #2710, #3162 | Dependabot: major versions of upload-artifact, download-artifact, setup-node, setup-uv | Defer | They change many CI workflows at once. #2706, #2710 and #3162 have failing checks. |
| Pipeline #381, #382 | Dependabot: attest actions | Defer | Build attestation only. #382 fails 2 checks on its old base. |
| WebApp #1139 | WebApp half of #3045: sites confirm task facts | Defer | Off by default until its Pipeline half lands (inside #3056). 4 conflicts. |
| WebApp #1088 | Robot-team setup in the ops job console, for 16 arm configurations | Defer | The beta runs Blueprint's own screening, not robots that teams set up. Its Pipeline half (#2747) merged on 10-21. 1 conflict and open review findings. |
| WebApp #931 | Luna-to-Haiku model swap (WebApp) | Defer | Your decision in the brief. `main` still runs `gpt-6-luna`. |
| WebApp #1129 | Test only: a cross-repo replay test that CI skips by default | Defer | It matches the Pipeline's AUT-04 behaviour (#3028, merged). Safe to merge at any time. |
| WebApp #1083 | Test only: the Pipeline's capability-gap example in the WebApp fixture | Defer | No live code sends that message. Safe to merge at any time. |
| WebApp #838 | Adds the Simplified Technical English skill for agents | Defer | Agent tooling, not product code. The Pipeline already has it. |

## To close the recommended 33

```bash
for n in 3042 3045 3052 3051 3037 2548 1980 265 266 267 268 269 56; do gh pr close "$n" --repo ognjhunt/BlueprintCapturePipeline --comment "Closed in the 2026-11-03 open PR triage (docs/launch-readiness/2026-11-03-open-pr-triage.md)."; done
```

```bash
for n in 1136 1085 918 915; do gh pr close "$n" --repo ognjhunt/Blueprint-WebApp --comment "Closed in the 2026-11-03 open PR triage (BlueprintCapturePipeline docs/launch-readiness/2026-11-03-open-pr-triage.md)."; done
```

```bash
for n in 93 54 32 31 30 28 27 25 24 23 21 18 16 15 14 13; do gh pr close "$n" --repo ognjhunt/BlueprintCapture --comment "Closed in the 2026-11-03 open PR triage (BlueprintCapturePipeline docs/launch-readiness/2026-11-03-open-pr-triage.md)."; done
```

## Done

11-03 10:50–10:52Z, on the owner's OK: the 33 PRs were closed with the comments
above, and none was merged. The 23 deferred PRs and the cloud session's PRs
(Pipeline #3196, WebApp #1198 and #1175) stay open.

Dependabot deleted the branches of #265, #266, #267, #269 and #56 when they
closed, as it does for its own PRs; #268's branch was still there at 10:55Z.
Every closed PR keeps its commits under `refs/pull/<number>/head`, and its page
can restore the branch.
