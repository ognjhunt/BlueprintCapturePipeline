"""Trusted headless Blender wrapper for a retained candidate authoring program.

Invoked only through asset_authoring_sandbox. Standardized studio renders are
visual review media, not calibrated scene observations or native qualification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def main(root: Path) -> None:
    import bpy
    from mathutils import Vector

    inputs = json.loads((root / 'render_inputs.json').read_text())
    dimensions = tuple(inputs['dimensions_m'])
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1.0
    bpy.ops.wm.stl_import(filepath=str(root / 'candidate.stl'))
    cad_base = bpy.context.active_object
    cad_base.name = 'CAD_BASE'
    for vertex in cad_base.data.vertices:
        vertex.co *= 0.001
    vertices = [v.co for v in cad_base.data.vertices]
    minimum = Vector(tuple(min(v[a] for v in vertices) for a in range(3)))
    maximum = Vector(tuple(max(v[a] for v in vertices) for a in range(3)))
    offset = Vector(((minimum.x + maximum.x) / 2, (minimum.y + maximum.y) / 2, minimum.z))
    for vertex in cad_base.data.vertices:
        vertex.co -= offset
    source_images = [bpy.data.images.load(str(root / name), check_existing=True)
                     for name in inputs['reference_files']]
    namespace = {'CAD_BASE': cad_base, 'DIMENSIONS': dimensions,
                 'SOURCE_IMAGES': source_images, '__name__': '__asset_candidate__'}
    code = (root / 'asset_program.py').read_text()
    exec(compile(code, str(root / 'asset_program.py'), 'exec'), namespace)  # nosec B102 - OS sandbox, intentional candidate execution
    bpy.context.view_layer.update()
    objects = [obj for obj in bpy.context.scene.objects
               if obj.type == 'MESH' and not obj.hide_render]
    if not objects:
        raise ValueError('blender_candidate_visible_geometry_missing')
    if any(obj.type not in {'MESH', 'EMPTY'} for obj in bpy.context.scene.objects):
        raise ValueError('blender_candidate_nonasset_object_forbidden')
    points = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        points.extend(evaluated.matrix_world @ vertex.co for vertex in mesh.vertices)
        evaluated.to_mesh_clear()
    minimum = [min(point[a] for point in points) for a in range(3)]
    maximum = [max(point[a] for point in points) for a in range(3)]
    materials = []
    seen = set()
    for obj in objects:
        if not obj.data.materials:
            raise ValueError('blender_candidate_mesh_material_missing:' + obj.name)
        for material in obj.data.materials:
            if material is None or not material.use_nodes:
                raise ValueError('blender_candidate_surface_nodes_missing')
            if material.name in seen:
                continue
            seen.add(material.name)
            outputs = [node for node in material.node_tree.nodes if node.type == 'OUTPUT_MATERIAL' and node.is_active_output]
            if len(outputs) != 1 or not outputs[0].inputs['Surface'].is_linked:
                raise ValueError('blender_candidate_surface_output_invalid')
            surface = outputs[0].inputs['Surface'].links[0].from_node
            if surface.type != 'BSDF_PRINCIPLED':
                raise ValueError('blender_candidate_nonpreview_surface')
            if surface.inputs['Alpha'].is_linked or surface.inputs['Transmission Weight'].is_linked:
                raise ValueError('blender_candidate_variable_optical_opacity_forbidden')
            materials.append({'name': material.name,
                              'alpha': surface.inputs['Alpha'].default_value,
                              'transmission': surface.inputs['Transmission Weight'].default_value,
                              'roughness': surface.inputs['Roughness'].default_value,
                              'source_texture_images': [node.image.name for node in material.node_tree.nodes
                                                        if node.type == 'TEX_IMAGE' and node.image]})
    readback = {'dimensions_m': [maximum[a] - minimum[a] for a in range(3)],
                'minimum_z_m': minimum[2],
                'center_xy_m': [(minimum[a] + maximum[a]) / 2 for a in (0, 1)],
                'minimum_xyz_m': minimum, 'maximum_xyz_m': maximum,
                'mesh_count': len(objects), 'vertex_count': len(points),
                'materials': materials, 'blender_version': bpy.app.version_string,
                'blender_build_hash': bpy.app.build_hash.decode(),
                'cad_import_recenter_offset_m': list(offset),
                'program_sha256': hashlib.sha256(code.encode()).hexdigest(),
                'render_claim': 'candidate_studio_visual_review_only'}
    (root / 'geometry_readback.json').write_text(json.dumps(readback, indent=2) + '\n')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    # Export the asset before adding studio cameras, lights, and the floor.
    bpy.ops.wm.usd_export(filepath=str(root / 'candidate.usdc'),
                          selected_objects_only=True, export_materials=True,
                          generate_preview_surface=True, export_textures_mode='NEW',
                          relative_paths=True, root_prim_path='/Asset')
    for image in source_images:
        image.pack()
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 960
    scene.render.resolution_y = 960
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = False
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.32, 0.32, 0.32, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.55
    scene.view_settings.view_transform = 'AgX'
    size = max(dimensions)
    target = Vector((0, 0, dimensions[2] / 2))
    for name, location, energy in (
        ('StudioKey', (size, -size, 2 * size), 100),
        ('StudioFill', (-size, size / 2, size), 50),
    ):
        light = bpy.data.lights.new(name, 'AREA')
        light.energy = energy * size * size
        light.shape = 'DISK'
        light.size = size * 2
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (target - obj.location).to_track_quat('-Z', 'Y').to_euler()
    camera = bpy.data.cameras.new('StudioCamera')
    camera.type = 'ORTHO'
    camera.ortho_scale = size * 1.4
    obj = bpy.data.objects.new('StudioCamera', camera)
    scene.collection.objects.link(obj)
    scene.camera = obj
    views = {'perspective': (size, -size * 1.3, size * 1.6),
             'top': (0, 0, size * 2), 'side': (size * 1.5, -size, size * .5)}
    for name, position in views.items():
        obj.location = position
        obj.rotation_euler = (target - obj.location).to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = str(root / f'{name}.png')
        bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(root / 'candidate.blend'))


if __name__ == '__main__':
    main(Path(sys.argv[sys.argv.index('--') + 1]).resolve(strict=True))
