import logging

import bpy
from bpy.types import Context

from .str_reader import Model, PrimitiveType


def import_str(context: Context, model: Model) -> None:
    for display_list in model.display_lists:
        if display_list.vertices.size == 0:
            return
        if "position" not in display_list.vertices.dtype.names:
            return

        triangles: list[tuple[int, int, int]] = []
        vertex_cursor = 0
        for prim in display_list.primitives:
            match prim.prim_type:
                case PrimitiveType.POINTS:
                    pass
                case PrimitiveType.TRIANGLE_STRIP:
                    pass
                case PrimitiveType.TRIANGLES:
                    pass
                case _:
                    logging.error(f"Unimplemented primitive type {prim.prim_type}")
            vertex_cursor += prim.vertex_count

        mesh = bpy.data.meshes.new(model.name)
        mesh.from_pydata(display_list.vertices["position"], [], triangles)

        mesh_obj = bpy.data.objects.new(model.name, mesh)
        context.collection.objects.link(mesh_obj)
