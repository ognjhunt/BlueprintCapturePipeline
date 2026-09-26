# G1 runtime dependency review — 2026-09-26

Status: **pending exact owner review**. These CPython 3.12 Linux x86_64 wheels are proposed for internal, `development_only` Unitree G1 simulation. The previous paid preflight verified that `pinocchio` and `google.protobuf` were absent before any model download or episode. The selected `pin==4.1.0` release depends on the CMeel/Coal closure below. No external model weights are included in these wheels.

The repository policy in `docs/runtime_dependency_license_policy.json` requires owner review for each new exact name/version before runtime use. This table records PyPI wheel bytes and their embedded license metadata. PyPI release metadata and the embedded wheel licenses should be inspected before approval. The Qhull wheel carries its own notice and redistribution conditions; the immutable wheel retains that notice.

| Package | Version | Embedded license | SHA-256 |
| --- | --- | --- | --- |
| `cmeel` | `0.61.0` | `BSD-2-Clause` | `924f04ac525c1e2c4c42963b3a18a423926d019f81a683dfbab18143721963a8` |
| `cmeel-assimp` | `6.0.5` | `BSD-3-Clause` | `7b5ee36f3027de9b1bd73bd4294b60fed1bbc6c1524f94f64f69f38c5a49bf03` |
| `cmeel-boost` | `1.90.0` | `BSL-1.0` | `c306b9af69d13502a6598672a341deb4f549fa3eb8cfb363de190e7409919a4a` |
| `cmeel-console-bridge` | `1.0.2.3` | `Zlib` | `5bb1115ed38441b2396e732e10ec63d1e68445674f9f5d321f7985eb10e9aeef` |
| `cmeel-octomap` | `1.10.0` | `BSD-3-Clause` | `b0b54fac180dce4f483afe7029c29cc55f6f2b21be8413e8e2275845b0c204d7` |
| `cmeel-qhull` | `8.0.2.1` | `Qhull` | `2371a7c80a14f3e874876359ae3e3094861f081fcdd7a03987c3e880d14e07b9` |
| `cmeel-tinyxml2` | `11.0.0` | `Zlib` | `18674156bd41f3993dc1d5199da04fa496674358daa6588090fb9f86c71917b0` |
| `cmeel-urdfdom` | `6.0.0` | `BSD-3-Clause` | `7ab1be680a8ec866d5422c617b641d1f0e38774061df28b8b426fb26edce6337` |
| `cmeel-zlib` | `1.3.2` | `Zlib` | `4658000c5531273d14ca8f0250abcd2a2ad85b336f10a273a77ecb38611a5f5a` |
| `coal` | `3.0.3` | `BSD-3-Clause` | `b98f45b92af8f6608a97d47f05abab861af13ba54a47c82dfc22c9f4e7754a18` |
| `eigenpy` | `3.13.0` | `BSD-2-Clause` | `cde8ed4080da67e0c037ba1453c8d33c74465714ce5196781b640dcc398579a5` |
| `libcoal` | `3.0.3` | `BSD-3-Clause` | `28fa1473d80728994f7275b8c8797858959ab77a643f07f2222feb6de155bb8e` |
| `libpinocchio` | `4.1.0` | `BSD-3-Clause` | `dbcc81e9e302ae57700775af0b86ed353effe47b5d33fa12aff9936ac33c0ae0` |
| `numpy` | `2.3.1` | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | `e7cbf5a5eafd8d230a3ce356d892512185230e4781a361229bd902ff403bc660` |
| `pin` | `4.1.0` | `BSD-3-Clause` | `5971393ade6d5f0577bb18e532b294a2d8ea1a7bd552d0bbc3b064368c8777ab` |
| `protobuf` | `6.33.6` | `BSD-3-Clause` | `77179e006c476e69bf8e8ce866640091ec42e1beb80b213c3900006ecfba6901` |

Review scope: internal G1 simulation inside a pinned provider image, immutable packet staging, no public media or physical deployment claim. The wheel archive and its license files are retained with the packet. The source packet verifier checks every wheel hash before GPU allocation; the provider verifies the packet and imports before policy checkpoint download.

Owner decision: pending. Do not treat the previous approval of the pi0.5 base model and NVIDIA SONIC weights as approval of this new dependency set.
