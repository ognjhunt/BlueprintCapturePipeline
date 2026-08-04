"""Parametric CAD source for the sealed ADP-009A canned-beverage control.

The publisher semantic OBB is appearance/identity evidence, not a physical
surface measurement.  Accordingly this is a dimension-matched control asset,
not a claim that the original can's detailed geometry was measured.

Coordinate convention:
- origin: center of the circular base datum
- XY: base plane
- +Z: can height
- units: millimetres (build123d/STEP convention)
"""

from build123d import Align, Cylinder


DIAMETER_MM = 62.18945243762791
HEIGHT_MM = 169.42799377441406


def gen_step():
    """Return one labeled, closed solid with its base centered at the origin."""

    body = Cylinder(
        radius=DIAMETER_MM / 2.0,
        height=HEIGHT_MM,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    body.label = "canned_beverage_body"
    return body
