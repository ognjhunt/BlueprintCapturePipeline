# Causal-conditioning development result

The inputs, thresholds, and original development analysis were frozen before
the 49-session label-free held-out diagnostic and before any Experiment 2
provider call. This publication copy corrects the validity-rate interval to
cluster by session after review; see `statistical_correction_amendment_002.json`.
It uses the 14 historical pilot and calibration sessions only; no held-out
outcome label was opened.

The visible OSCAR skeleton annotation has a strong action-alignment signal, but
the generated scene after conservatively masking that annotation does not pass
the frozen action-following gates.

| Partition | Channel | Mean excess over strongest placebo | Clustered 95% interval | Session-cluster mean pass rate | Cluster-bootstrap lower 95% |
|---|---|---:|---:|---:|---:|
| Pilot | full generated crop | 0.28589 | [0.26292, 0.31303] | 0.83673 | 0.73469 |
| Pilot | skeleton-overlay region | 0.29285 | [0.26914, 0.32014] | 0.87755 | 0.79592 |
| Pilot | overlay-masked residual | 0.05964 | [0.02642, 0.09078] | 0.51020 | 0.42857 |
| Calibration | full generated crop | 0.26255 | [0.23932, 0.29140] | 0.81633 | 0.73469 |
| Calibration | skeleton-overlay region | 0.27094 | [0.24706, 0.29791] | 0.83673 | 0.75510 |
| Calibration | overlay-masked residual | 0.01865 | [-0.06123, 0.10148] | 0.44898 | 0.26531 |

The pooled 14-session residual standard deviation is 0.09002. With 49
independent held-out session clusters, the one-sided normal-approximation
minimum detectable excess is 0.03198, below the frozen meaningful margin of
0.05. The held-out causal diagnostic is therefore adequately powered for that
margin and is admitted without changing any threshold.

This diagnostic can falsify temporal action alignment. It cannot prove
counterfactual causality because OSCAR did not release alternate zero/shuffled
regenerations for these sessions, and the unavailable per-session camera
calibration prevents a defensible reconstruction of those conditioning videos.
