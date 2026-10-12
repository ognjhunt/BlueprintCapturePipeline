"""Source checks on final upstream geometry, without a geometry author/planner."""
from __future__ import annotations

import itertools
from .task_object_authoring_contracts import require
from .task_object_cavity_validation import collision_cavity_findings


def collision_piece(collider, link, transforms):
    from pxr import Gf, UsdGeom, UsdPhysics
    matrix = transforms.GetLocalToWorldTransform(collider) * transforms.GetLocalToWorldTransform(link).GetInverse()
    if collider.IsA(UsdGeom.Cube):
        half = float(UsdGeom.Cube(collider).GetSizeAttr().Get()) / 2
        points = list(itertools.product((-half, half), repeat=3))
        quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
        faces = [i for q in quads for triangle in ((q[0], q[1], q[2]), (q[0], q[2], q[3])) for i in triangle]
        approximation = "analytic_box"
    elif collider.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(collider)
        points = list(mesh.GetPointsAttr().Get() or [])
        indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        faces, offset = [], 0
        for count in mesh.GetFaceVertexCountsAttr().Get() or []:
            face = indices[offset:offset + count]
            faces.extend(i for index in range(1, count - 1) for i in (face[0], face[index], face[index + 1]))
            offset += count
        approximation = str(UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get())
    else:
        return {"approximation": "analytic_" + collider.GetTypeName(), "vertices": [], "faces": []}
    return {"approximation": approximation, "vertices": [list(matrix.Transform(Gf.Vec3d(*p))) for p in points], "faces": faces}


def validate_cavities(stage, annotations, request, bodies, colliders, transforms, owner):
    from pxr import UsdPhysics
    configuration = request["configuration"]
    cavities = annotations.get("interior_cavities")
    require(annotations.get("schema_version") == "artisan_candidate_annotations.v1"
            and annotations.get("request_digest") == request["request_digest"]
            and isinstance(cavities, list) and cavities, "artisan_cavity_annotations_required")
    retained = configuration.get("interior_cavities")
    require(retained is None or cavities == retained, "artisan_retained_cavity_constraints_changed")
    links = {body.GetName(): body for body in bodies}
    observations = []
    for row in cavities:
        require(row.get("link_id") in links and row.get("basis") in {"generated_candidate", "source_constraint"},
                "artisan_cavity_annotation_identity_invalid")
        require(len(row.get("opening_center_link_m") or []) == 3
                and all(isinstance(row.get(key), (int, float)) and row[key] > 0
                        for key in ("opening_width_m", "opening_height_m", "inner_depth_m")),
                "artisan_cavity_probe_region_invalid")
        body = links[row["link_id"]]
        pieces = [collision_piece(p, body, transforms) for p in colliders if owner(p, UsdPhysics) == body]
        findings = collision_cavity_findings(pieces, [row])
        require(not findings, "artisan_cooked_collision_cavity_unproven:" + ";".join(findings))
        observations.append({"cavity": row, "findings": [], "basis": row["basis"],
                             "native_collision_cooking_still_required": True})
    return observations
