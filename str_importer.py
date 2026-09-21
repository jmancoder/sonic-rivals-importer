import logging

import bpy
from bpy.types import Context, Object
import numpy as np
import numpy.typing as npt

from . import str_reader

logger = logging.getLogger(__name__)


def _import_submeshes(context, mesh_data: str_reader.Mesh) -> list[Object]:
    mesh_objects: list[Object] = []
    for i, display_list in enumerate(mesh_data.display_lists):
        if display_list.vertices.size == 0:
            logging.warning("Display list %d contained no vertices", i)
            continue
        if "position" not in display_list.vertices.dtype.names:
            logging.warning("Display list %d contained no vertex positions", i)
            continue

        # Convert primitives to triangles
        triangles: list[tuple[int, int, int]] = []
        vertex_idx = 0
        for prim in display_list.primitives:
            match prim.prim_type:
                case str_reader.PrimitiveType.POINTS:
                    pass
                case str_reader.PrimitiveType.TRIANGLES:
                    for i in range(0, prim.vertex_count - 2, 3):
                        triangles.append(
                            (
                                vertex_idx + i,
                                vertex_idx + i + 1,
                                vertex_idx + i + 2,
                            )
                        )
                case str_reader.PrimitiveType.TRIANGLE_STRIP:
                    for i in range(prim.vertex_count - 2):
                        if i & 1:
                            triangles.append(
                                (
                                    vertex_idx + i,
                                    vertex_idx + i + 2,
                                    vertex_idx + i + 1,
                                )
                            )
                        else:
                            triangles.append(
                                (
                                    vertex_idx + i,
                                    vertex_idx + i + 1,
                                    vertex_idx + i + 2,
                                )
                            )
                case str_reader.PrimitiveType.TRIANGLE_FAN:
                    for i in range(1, prim.vertex_count - 1):
                        triangles.append(
                            (vertex_idx, vertex_idx + i, vertex_idx + i + 1)
                        )
                case _:
                    logger.error("Unimplemented primitive type %s", prim.prim_type)
            vertex_idx += prim.vertex_count

        # Quantize positions
        positions = display_list.vertices["position"]
        if display_list.vertex_flags.position_format == 1:
            positions = positions.astype(float) / 127.0
        elif display_list.vertex_flags.position_format == 2:
            positions = positions.astype(float) / 32767.0

        # Import geometry
        mesh = bpy.data.meshes.new("Mesh")
        mesh.from_pydata(positions, [], triangles)

        # Remove degenerate triangles
        mesh.validate(verbose=True)
        mesh.update()

        # Quantize and import normals
        if "normal" in display_list.vertices.dtype.names:
            normals = display_list.vertices["normal"]
            if display_list.vertex_flags.normal_format == 1:
                normals = normals.astype(float) / 127.0
            elif display_list.vertex_flags.normal_format == 2:
                normals = normals.astype(float) / 32767.0
            mesh.normals_split_custom_set_from_vertices(normals)

        # Quantize and import UVs
        if "uv" in display_list.vertices.dtype.names:
            uvs = display_list.vertices["uv"]
            if display_list.vertex_flags.uv_format == 1:
                uvs = uvs.astype(float) / 127.0
            elif display_list.vertex_flags.normal_format == 2:
                uvs = uvs.astype(float) / 32767.0

            uv_layer = mesh.uv_layers.new()
            vertex_idx_array = np.empty(len(mesh.loops), dtype=np.int32)
            mesh.loops.foreach_get("vertex_index", vertex_idx_array)
            uv_layer.uv.foreach_set("vector", uvs[vertex_idx_array].ravel())

        # Convert and import vertex colors
        if "color" in display_list.vertices.dtype.names:
            raw_colors = display_list.vertices["color"]
            rgba = np.empty((raw_colors.size, 4), dtype=np.uint8)
            if display_list.vertex_flags.color_format == 4:
                # RGB565
                rgba[:, 0] = (raw_colors & 0x1F) * 255 // 31
                rgba[:, 1] = ((raw_colors >> 5) & 0x3F) * 255 // 63
                rgba[:, 2] = ((raw_colors >> 11) & 0x1F) * 255 // 31
                rgba[:, 3] = 255
            elif display_list.vertex_flags.color_format == 5:
                # RGBA5551
                rgba[:, 0] = (raw_colors & 0x1F) * 255 // 31
                rgba[:, 1] = ((raw_colors >> 5) & 0x1F) * 255 // 31
                rgba[:, 2] = ((raw_colors >> 10) & 0x1F) * 255 // 31
                rgba[:, 3] = ((raw_colors >> 15) & 0x1) * 255
            elif display_list.vertex_flags.color_format == 6:
                # RGBA4444
                rgba[:, 0] = (raw_colors & 0xF) * 0x11
                rgba[:, 1] = ((raw_colors >> 4) & 0xF) * 0x11
                rgba[:, 2] = ((raw_colors >> 8) & 0xF) * 0x11
                rgba[:, 3] = ((raw_colors >> 12) & 0xF) * 0x11
            else:
                # RGBA8888
                rgba = np.ascontiguousarray(raw_colors).view(np.uint8)
                pass

            vertex_color_attr = mesh.color_attributes.new(
                name="vertex_color",
                type="BYTE_COLOR",
                domain="POINT",
            )
            vertex_color_attr.data.foreach_set(
                "color",
                rgba.ravel(),
            )

        # Create mesh object
        mesh_obj = bpy.data.objects.new("Mesh", mesh)
        context.collection.objects.link(mesh_obj)
        mesh_objects.append(mesh_obj)
    return mesh_objects


def import_str(context: Context, model: str_reader.Model) -> None:
    for mesh_data in model.meshes:
        _import_submeshes(context, mesh_data)
