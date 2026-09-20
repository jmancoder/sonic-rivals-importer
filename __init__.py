bl_info = {
    "name": "ADDON_NAME",
    "author": "AUTHOR_NAME",
    "description": "",
    "blender": (2, 80, 0),
    "version": (0, 0, 1),
    "location": "File > Import",
    "warning": "",
    "category": "Import-Export",
}

from pathlib import Path

import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty
from bpy.types import Operator, Context

from . import str_reader


class IMPORT_OT_SCENE_str(Operator, ImportHelper):
    """Load an STR file."""

    bl_idname = "import_scene.sonic_rivals_str"
    bl_label = "Import STR"
    filename_ext = ".str"

    filter_glob: StringProperty(
        default="*.str",
        options={"HIDDEN"},
        maxlen=255,
    )

    def execute(self, context: Context):
        in_path = Path(self.filepath)
        with open(in_path, "rb") as f:
            str_reader.read_str(f)
        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(
        IMPORT_OT_SCENE_str.bl_idname, text="Sonic Rivals Stream (.str)"
    )


def register():
    bpy.utils.register_class(IMPORT_OT_SCENE_str)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.utils.unregister_class(IMPORT_OT_SCENE_str)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    register()
