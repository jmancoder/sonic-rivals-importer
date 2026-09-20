import logging

import bpy
from bpy.types import Context

from .str_reader import Model, PrimitiveType

logger = logging.getLogger(__name__)


def import_str(context: Context, model: Model) -> None:
    for display_list in model.display_lists:
        if display_list.vertices.size == 0:
            return
        if "position" not in display_list.vertices.dtype.names:
            return

        # Convert primitives to triangles
        triangles: list[tuple[int, int, int]] = []
        vertex_idx = 0
        for prim in display_list.primitives:
            match prim.prim_type:
                case PrimitiveType.POINTS:
                    pass
                case PrimitiveType.TRIANGLES:
                    for i in range(0, prim.vertex_count - 2, 3):
                        triangles.append(
                            (
                                vertex_idx + i,
                                vertex_idx + i + 1,
                                vertex_idx + i + 2,
                            )
                        )
                case PrimitiveType.TRIANGLE_STRIP:
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
                case _:
                    logger.error("Unimplemented primitive type %s", prim.prim_type)
            vertex_idx += prim.vertex_count

        # Quantize positions
        positions = display_list.vertices["position"]
        if display_list.vertex_flags.position_format == 1:
            positions = positions.astype(float) / 127.0
        elif display_list.vertex_flags.position_format == 2:
            positions = positions.astype(float) / 32767.0

        mesh = bpy.data.meshes.new(model.name)
        mesh.from_pydata(positions, [], triangles)
        mesh.validate(verbose=True)
        mesh.update()

        mesh_obj = bpy.data.objects.new(model.name, mesh)
        context.collection.objects.link(mesh_obj)
