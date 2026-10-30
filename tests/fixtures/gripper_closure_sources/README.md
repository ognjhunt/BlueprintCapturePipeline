Retained publisher URDFs and only the finger collision meshes used by the closure regression. These fixtures preserve their original bytes, pinned commit identifiers, SHA-256 hashes and publisher licenses in sources.json. They establish joint/mesh geometry, not native simulation, collision solver or physical grasp performance. Other URDF mesh references are intentionally absent and these partial fixtures cannot be imported as complete robots.

The regression transforms the source collision vertices through the source
prismatic joint origins and axes and compares the distal 20 mm envelopes.
TRLC at q=0 has inner pad Y coordinates approximately +0.0400203 / -0.0400203 m;
negative joint travel closes the jaws. Its full -0.045 m command overlaps those
envelopes by about 10 mm, so this is a closing-direction correction, not a
collision-free closed-pose assertion. Axol at q=0 has inner distal pad X
coordinates approximately -0.00094 / +0.00094 m; q=-0.044 m opens the gap by
0.088 m. Both arms use that convention. Nominal gripper widths and contact
performance remain simulation declarations pending native measurement.
