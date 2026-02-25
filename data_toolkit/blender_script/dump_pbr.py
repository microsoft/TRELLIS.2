import argparse, sys, os, math, io
from typing import *
import bpy
import bmesh
from mathutils import Vector, Matrix
import numpy as np
from PIL import Image
import pickle


"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj if bpy.app.version[0] < 4 else bpy.ops.wm.obj_import,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl if bpy.app.version[0] < 4 else bpy.ops.wm.stl_import,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply if bpy.app.version[0] < 4 else bpy.ops.wm.ply_import,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}


def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS', bone_heuristic='TEMPERANCE')
    else:
        import_function(filepath=object_path)
        
        
def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    to_remove = []
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            to_remove.append(obj)
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)
      

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    
    return scale, offset


# =============== NODE TREE PARSING ===============

def extract_image(tex_node, channels):
        image = tex_node.image
        pixels = np.array(image.pixels[:])
        data = pixels.reshape(image.size[1], image.size[0], -1)
        data = data[..., channels]

        if data.dtype != np.uint8:
            data = np.clip(data, 0.0, 1.0)
            data = (data * 255).astype(np.uint8)

        if len(data.shape) == 2:  # Single channel
            pil_image = Image.fromarray(data, mode='L')
        elif data.shape[2] == 3:
            pil_image = Image.fromarray(data, mode='RGB')
        elif data.shape[2] == 4:
            pil_image = Image.fromarray(data, mode='RGBA')
        else:
            raise ValueError("Unsupported channel shape for image")

        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        png_bytes = buffer.getvalue()

        return {
            'image': png_bytes,
            'interpolation': tex_node.interpolation,
            'extension': tex_node.extension,
        }


def try_extract_image(link, expected_channel='RGB'):
    """
    Tries to extract an image from a texture node link.
    Supported sub tree modes:
      - RGB:
        TEX_IMAGE ->
      - R, G, B:
        TEX_IMAGE -> SEPARATE_COLOR ->
      - A:
        TEX_IMAGE ->
    """
    assert expected_channel in ['RGB', 'R', 'G', 'B', 'A'], "Unsupported channel"

    if expected_channel == 'RGB':
        assert link.from_node.type == 'TEX_IMAGE', "Material is not supported"
        assert link.from_socket.name == 'Color', "Material is not supported"
        tex_node = link.from_node
        return extract_image(tex_node, [0, 1, 2])

    if expected_channel in ['R', 'G', 'B']:
        socket_name = {
            'R': 'Red',
            'G': 'Green',
            'B': 'Blue',
        }[expected_channel]
        assert link.from_node.type == 'SEPARATE_COLOR' and link.from_node.mode == 'RGB', \
            f"Material is not supported, {link.from_node.type}, {link.from_node.mode}"
        assert link.from_socket.name == socket_name, "Material is not supported"
        sep_node = link.from_node
        assert sep_node.inputs[0].is_linked and sep_node.inputs[0].links[0].from_node.type == 'TEX_IMAGE', \
            "Material is not supported"
        assert sep_node.inputs[0].links[0].from_socket.name == 'Color', "Material is not supported"
        tex_node = sep_node.inputs[0].links[0].from_node
        channel_index = {
            'R': 0,
            'G': 1,
            'B': 2,
        }[expected_channel]
        return extract_image(tex_node, channel_index)

    if expected_channel == 'A':
        assert link.from_node.type == 'TEX_IMAGE', "Material is not supported"
        assert link.from_socket.name == 'Alpha', "Material is not supported"
        tex_node = link.from_node
        return extract_image(tex_node, 3)


def try_extract_factor(link, mode='color'):
    """
    Tries to extract a factor from a math node link.
    Supported sub tree modes:
      - color:
       ANY -> MIX(MULTIPLY) ->
      - scalar:
       ANY -> MATH(MULTIPLY) ->
    """
    assert mode in ['color','scalar'], "Unsupported mode"

    if mode == 'color':
        if link.from_node.type == 'MIX':
            mix_node = link.from_node
            assert mix_node.data_type == 'RGBA' and mix_node.blend_type == 'MULTIPLY', f"Material is not supported, {mix_node.data_type}, {mix_node.blend_type}"
            assert not mix_node.inputs['Factor'].is_linked and mix_node.inputs['Factor'].default_value == 1.0, \
                "Material is not supported"
            if mix_node.inputs['A'].is_linked:
                assert not mix_node.inputs['B'].is_linked, "Material is not supported"
                return (list(mix_node.inputs['B'].default_value)[:3], mix_node.inputs['A'].links[0])
            else:
                assert not mix_node.inputs['A'].is_linked, "Material is not supported"
                assert mix_node.inputs['B'].is_linked, "Material is not supported"
                return (list(mix_node.inputs['A'].default_value)[:3], mix_node.inputs['B'].links[0])
        return ([1.0, 1.0, 1.0], link)

    if mode =='scalar':
        if link.from_node.type == 'MATH':
            math_node = link.from_node
            assert math_node.operation == 'MULTIPLY', "Material is not supported"
            assert math_node.inputs[0].is_linked, "Material is not supported"
            assert not math_node.inputs[1].is_linked, "Material is not supported"
            return (math_node.inputs[1].default_value, math_node.inputs[0].links[0])
        return (1.0, link)


def try_extract_image_with_factor(link, expected_channel='RGB'):
    """
    Tries to extract an image and a factor from a texture node link.
    """
    factor, link = try_extract_factor(link, 'color' if expected_channel in ['RGB'] else 'scalar')
    image = try_extract_image(link, expected_channel)
    return (factor, image)


def main(arg):    
    # Initialize context
    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
    print('[INFO] Scene initialized.')
    
    # Normalize scene
    scale, offset = normalize_scene()
    print('[INFO] Scene normalized.')
    
    # Start dumping
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    output = {
        'materials': [],
        'objects': [],
    }

    # Dumping materials
    for mat in bpy.data.materials:
        assert mat.use_nodes == True, "Material is not supported"

        pack = {
            "baseColorFactor": [1.0, 1.0, 1.0],
            "alphaFactor": 1.0,
            "metallicFactor": 1.0,
            "roughnessFactor": 1.0,
            "alphaMode": "OPAQUE",
            "alphaCutoff": 0.5,
            "baseColorTexture": None,
            "alphaTexture": None,
            "metallicTexture": None,
            "roughnessTexture": None,
        }

        try:
            principled_node = mat.node_tree.nodes.get('Principled BSDF')
            assert principled_node is not None, "Material is not supported"

            # Handle base color
            if not principled_node.inputs['Base Color'].is_linked:
                pack["baseColorFactor"] = list(principled_node.inputs['Base Color'].default_value)
            else:
                link = principled_node.inputs['Base Color'].links[0]
                if link.from_node.type == 'RGB':
                    pack["baseColorFactor"] = list(link.from_node.outputs[0].default_value)
                else:
                    factor, image = try_extract_image_with_factor(link, 'RGB')
                    pack["baseColorFactor"] = factor
                    pack["baseColorTexture"] = image

            # Handle alpha
            if not principled_node.inputs['Alpha'].is_linked:
                pack["alphaFactor"] = principled_node.inputs['Alpha'].default_value
                if pack["alphaFactor"] < 1.0:
                    pack["alphaMode"] = "BLEND"
            else:
                link = principled_node.inputs['Alpha'].links[0]
                node = link.from_node
                if node.type == 'VALUE':
                    pack["alphaFactor"] = node.outputs[0].default_value
                    if pack["alphaFactor"] < 1.0:
                        pack["alphaMode"] = "BLEND"
                else:
                    pack["alphaMode"] = "BLEND"
                    if node.type == 'MATH':
                        if node.operation == 'ROUND':
                            assert node.inputs[0].is_linked, "Material is not supported"
                            pack["alphaMode"] = "MASK"
                            link = node.inputs[0].links[0]
                        elif node.operation == 'SUBTRACT':
                            assert node.inputs[0].default_value == 1.0 and \
                                node.inputs[1].is_linked and \
                                node.inputs[1].links[0].from_node.type == 'MATH' and \
                                node.inputs[1].links[0].from_node.operation == 'LESS_THAN', \
                                "Material is not supported"
                            assert node.inputs[1].links[0].from_node.inputs[0].is_linked, "Material is not supported"
                            pack["alphaMode"] = "MASK"
                            pack["alphaCutoff"] = node.inputs[1].links[0].from_node.inputs[1].default_value
                            link = node.inputs[1].links[0].from_node.inputs[0].links[0]
                    factor, image = try_extract_image_with_factor(link, 'A')
                    pack["alphaFactor"] = factor
                    pack["alphaTexture"] = image

            # Handle metallic
            if not principled_node.inputs['Metallic'].is_linked:
                pack["metallicFactor"] = principled_node.inputs['Metallic'].default_value
            else:
                link = principled_node.inputs['Metallic'].links[0]
                node = link.from_node
                if node.type == 'VALUE':
                    pack["metallicFactor"] = node.outputs[0].default_value
                else:
                    factor, image = try_extract_image_with_factor(link, 'B')
                    pack["metallicFactor"] = factor
                    pack["metallicTexture"] = image

            # Handle roughness
            if not principled_node.inputs['Roughness'].is_linked:
                pack["roughnessFactor"] = principled_node.inputs['Roughness'].default_value
            else:
                link = principled_node.inputs['Roughness'].links[0]
                node = link.from_node
                if node.type == 'VALUE':
                    pack["roughnessFactor"] = node.outputs[0].default_value
                else:
                    factor, image = try_extract_image_with_factor(link, 'G')
                    pack["roughnessFactor"] = factor
                    pack["roughnessTexture"] = image

            output['materials'].append(pack)
        except Exception:
            with open(arg.output_path + '_error.txt', 'w') as f:
                f.write(str([[n.name] for n in mat.node_tree.nodes]))
            raise RuntimeError("Material is not supported")

    # Dumping meshes
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
        
        pack = {
            "vertices": None,
            "faces": None,
            "uvs": None,
            "matIDs": None,
        }
        
        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        
        bm = bmesh.new()
        bm.from_mesh(eval_mesh)
        bm.transform(obj.matrix_world)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.to_mesh(eval_mesh)
        bm.free()
                
        pack["vertices"] = np.array([
            v.co[:] for v in eval_mesh.vertices
        ], dtype=np.float32)   # (N, 3)
        
        pack["faces"] = np.array([
            [eval_mesh.loops[i].vertex_index for i in poly.loop_indices]
            for poly in eval_mesh.polygons
        ], dtype=np.int32)   # (F, 3)
        
        pack["normals"] = np.array([
            [eval_mesh.loops[i].normal for i in poly.loop_indices]
            for poly in eval_mesh.polygons
        ], dtype=np.float32)  # (F, 3, 3)
        
        if eval_mesh.uv_layers.active is not None:
            pack["uvs"] = np.array([
                [eval_mesh.uv_layers.active.data[i].uv for i in poly.loop_indices]
                for poly in eval_mesh.polygons
            ], dtype=np.float32)  # (F, 3, 2)

        pack["mat_ids"] = np.array([
            bpy.data.materials.find(obj.material_slots[poly.material_index].name)
            if len(obj.material_slots) > 0 and obj.material_slots[poly.material_index].material is not None else -1
            for poly in eval_mesh.polygons
        ], dtype=np.int32)

        output['objects'].append(pack)

    # Save output
    os.makedirs(os.path.dirname(arg.output_path), exist_ok=True)
    with open(arg.output_path, 'wb') as f:
        pickle.dump(output, f)
    print('[INFO] Output saved to {}.'.format(arg.output_path))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_path', type=str, default='/tmp', help='The path the output will be dumped to.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
    