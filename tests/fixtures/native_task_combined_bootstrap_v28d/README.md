V28d saved bootstrap evidence (2026-09-11 UTC)
=============================================

The provisioning JSON is the exact, unmodified 27,088-byte receipt retained after
Vast instance 50535878 terminated. Both child outputs reported
`native_task_isaaclab_provisioning_receipt_invalid` before native startup because
the receipt existed only in the parent output directory.

Receipt file SHA-256: `c044206784cf600a782896ea89302e234d723964a390e910fec05f07649e64ee`.
The three Isaac Lab experience files are exact bytes read from the retained
source archive `b7bbb7ae3064ad581a710d57121131eca776aabcab3ce2593b4ef396c6a3d3f1`.
Their original NVIDIA copyright/license headers remain intact.

The test redirects only original source-path resolution to these retained files,
then invokes the unchanged production launcher verifier. It never rewrites or
reseals the provisioning JSON. Simulator edges are CPU fakes; passing proves
bootstrap handoffs and program flow, not native render or physics behavior.
