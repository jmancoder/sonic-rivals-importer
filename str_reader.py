from dataclasses import dataclass
from enum import Enum
from io import BufferedReader
import logging
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from .binary_reader import BinaryReader, GECommand

logger = logging.getLogger(__name__)


class Material(NamedTuple): ...


class VertexFlags(NamedTuple):
    transform_bypass: bool
    morph_count: int
    weight_count: int
    index_format: int
    weight_format: int
    position_format: int
    normal_format: int
    color_format: int
    uv_format: int


class PrimitiveType(Enum):
    POINTS = 0
    LINES = 1
    LINE_STRIP = 2
    TRIANGLES = 3
    TRIANGLE_STRIP = 4
    TRIANGLE_FAN = 5
    SPRITES = 6


class Primitive(NamedTuple):
    prim_type: PrimitiveType
    vertex_count: int
    flags: int


class DisplayListInfo(NamedTuple):
    dlist_offset: int
    vertex_offset: int


@dataclass
class DisplayList:
    primitives: list[Primitive]
    vertices: npt.NDArray
    vertex_flags: VertexFlags
    vertex_off: int


class Mesh(NamedTuple):
    material: Material
    display_lists: list[DisplayList]


class Model(NamedTuple):
    name: str
    meshes: list[Mesh]


def _read_material(bs: BinaryReader) -> Material:
    sig = bs.read_int32()
    return Material()


def _vtype_flags_to_dtype(flags: VertexFlags) -> npt.DTypeLike:
    unsigned_types = {
        1: "<u1",
        2: "<u2",
        3: "<f4",
    }
    signed_types = {
        1: "<i1",
        2: "<i2",
        3: "<f4",
    }
    type_sizes = {
        1: 1,
        2: 2,
        3: 4,
    }

    def align(offset: int, alignment: int) -> int:
        return (offset + alignment - 1) & ~(alignment - 1)

    names = []
    formats = []
    offsets = []
    offset = 0
    vertex_alignment = 1
    if flags.weight_format:
        names.append("weights")
        formats.append((unsigned_types[flags.weight_format], flags.weight_count))
        offsets.append(offset)
        offset += type_sizes[flags.weight_format] * flags.weight_count
        vertex_alignment = max(
            vertex_alignment,
            type_sizes[flags.weight_format],
        )

    if flags.uv_format:
        alignment = type_sizes[flags.uv_format]
        offset = align(offset, alignment)
        names.append("uv")
        formats.append((unsigned_types[flags.uv_format], 2))
        offsets.append(offset)
        offset += type_sizes[flags.uv_format] * 2
        vertex_alignment = max(vertex_alignment, alignment)

    if flags.color_format:
        color_size = 2 if flags.color_format < 7 else 4
        offset = align(offset, color_size)
        names.append("color")
        formats.append("<u2" if color_size == 2 else "<u4")
        offsets.append(offset)
        offset += color_size
        vertex_alignment = max(vertex_alignment, color_size)

    if flags.normal_format:
        alignment = type_sizes[flags.normal_format]
        offset = align(offset, alignment)
        names.append("normal")
        formats.append((signed_types[flags.normal_format], 3))
        offsets.append(offset)
        offset += type_sizes[flags.normal_format] * 3
        vertex_alignment = max(vertex_alignment, alignment)

    if flags.position_format:
        alignment = type_sizes[flags.position_format]
        offset = align(offset, alignment)
        names.append("position")
        formats.append((signed_types[flags.position_format], 3))
        offsets.append(offset)
        offset += type_sizes[flags.position_format] * 3
        vertex_alignment = max(vertex_alignment, alignment)

    vertex_size = align(offset, vertex_alignment)
    vertex_dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": vertex_size,
        }
    )
    if flags.morph_count == 1:
        return vertex_dtype
    return np.dtype(
        [
            ("morphs", vertex_dtype, flags.morph_count),
        ]
    )


def _read_geometry(bs: BinaryReader) -> list[DisplayList]:
    geometry_off = bs.tell()
    logger.debug("Geometry offset: 0x%X", geometry_off)
    bs.read_int32()
    bs.read_int32()
    bs.read_int32()
    bs.read_int32()
    bs.read_vec4f()
    display_list_chunk_size = bs.read_uint32()
    display_list_count = bs.read_int32()
    vertex_start_off = geometry_off + display_list_chunk_size
    logger.debug("Display list count: %d", display_list_count)

    # Read display list offsets
    dlist_info_entries: list[DisplayListInfo] = []
    for _ in range(display_list_count):
        logger.debug("\nReading display list offset entry at 0x%X", bs.tell())
        vertices_off = 0
        while bs.tell() < vertex_start_off:
            unk_value = bs.read_int32()
            if unk_value == -1:
                break
            elif unk_value < 0:
                vertices_off = geometry_off + (~unk_value & 0xFFFFFFFF)
                break
            logger.debug(
                "Skipped unknown render value %i at 0x%X", unk_value, bs.tell() - 4
            )
        dlist_info_entries.append(
            DisplayListInfo(geometry_off + bs.read_uint32(), vertices_off)
        )

    # Read PSP GE commands
    display_lists: list[DisplayList] = []
    for dlist_info in dlist_info_entries:
        # Find VTYPE command
        bs.seek(dlist_info.dlist_offset)
        logger.debug("Reading display list GE commands at 0x%X", bs.tell())
        while True:
            command, argument = bs.read_ge_command()
            if (
                bs.tell() >= vertex_start_off
                or command == GECommand.VTYPE
                or command == GECommand.RET
            ):
                break
            logger.debug("Skipped GE command %s at 0x%X", command, bs.tell() - 4)
        if command != GECommand.VTYPE:
            if command == GECommand.RET:
                logger.error("Returned without finding VTYPE GE command")
            else:
                logger.error("Failed to find VTYPE or RET GE command")
            continue

        # Read vertex layout
        vertex_flags = VertexFlags(
            ((argument >> 23) & 0x1) > 0,
            ((argument >> 18) & 0x7) + 1,
            ((argument >> 14) & 0x7) + 1,
            (argument >> 11) & 0x3,
            (argument >> 9) & 0x3,
            (argument >> 7) & 0x3,
            (argument >> 5) & 0x3,
            (argument >> 2) & 0x7,
            argument & 0x3,
        )
        vertex_dtype = _vtype_flags_to_dtype(vertex_flags)

        # Read display list
        display_list = DisplayList(
            [], np.array([]), vertex_flags, dlist_info.vertex_offset
        )
        list_vertex_count = 0
        while True:
            command, argument = bs.read_ge_command()
            if command == GECommand.RET:
                break
            if command != GECommand.PRIM:
                logger.warning(
                    "Unimplemented GE command %s at 0x%X", command, bs.tell() - 4
                )
                continue
            flags = (argument & 0xF80000) >> 0x13
            primitive_type = PrimitiveType((argument & 0x70000) >> 0x10)
            prim_vertex_count = argument & 0xFFFF
            list_vertex_count += prim_vertex_count
            display_list.primitives.append(
                Primitive(primitive_type, prim_vertex_count, flags)
            )
        display_list.vertices = np.empty(list_vertex_count, vertex_dtype)
        display_lists.append(display_list)

    # Read vertices
    logger.debug("Vertex start offset: 0x%X", bs.tell())
    for display_list in display_lists:
        if display_list.vertex_off > 0:
            bs.seek(display_list.vertex_off)
        display_list.vertices[...] = np.frombuffer(
            bs.getbuffer(),
            display_list.vertices.dtype,
            display_list.vertices.size,
            bs.tell(),
        )
        bs.seek(display_list.vertices.nbytes, 1)
    return display_lists


def _read_model(bs: BinaryReader, base: int, models: list[Model]) -> None:
    logger.debug("Model offset: 0x%X", bs.tell())
    sig = bs.read_int32()
    next_ptr = bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_vec4f()
    bs.read_vec4f()
    bs.read_vec3f()
    bs.read_uint32()
    bs.read_vec3f()
    bs.read_uint32()
    bs.read_float()
    name_off = bs.read_uint32() - base
    unk_count = bs.read_int32()
    mesh_count = bs.read_int32()

    # Read meshes
    meshes: list[Mesh] = []
    for _ in range(mesh_count):
        material_off = bs.read_uint32() - base
        geometry_off = bs.read_uint32() - base
        next_mesh_off = bs.tell()

        bs.seek(material_off)
        material = _read_material(bs)
        bs.seek(geometry_off)
        geometry = _read_geometry(bs)
        meshes.append(Mesh(material, geometry))

        bs.seek(next_mesh_off)

    bs.seek(name_off)
    name = bs.read_aligned_cstring()

    # Read next model in linked list
    models.append(Model(name, meshes))
    if next_ptr > 0:
        bs.seek(next_ptr - base - 4)
        _read_model(bs, base, models)


def read_str_1(f: BufferedReader) -> list[Model]:
    bs = BinaryReader(f.read())

    bs.seek(0x1C)
    racer_type = bs.read_int32()
    if racer_type == 14:
        base = 0x8E149A0
    elif racer_type == 15:
        base = 0x9047CE0
    else:
        raise ValueError(f"Unexpected racer file type {racer_type}")

    bs.seek(0x7690)
    file_off_0 = bs.read_uint32() - base

    bs.seek(file_off_0)
    logger.debug("Layer 1 offset: 0x%X", bs.tell())
    flags = bs.read_uint32()
    unk_off_0 = bs.read_uint32() - base
    bs.seek(80, 1)
    unk_off_1 = bs.read_uint32() - base
    bs.seek(8, 1)
    unk_off_2 = bs.read_uint32() - base
    file_off_1 = bs.read_uint32() - base
    logger.debug(
        "Unknown offsets (layer 1): 0x%X, 0x%X, 0x%X", unk_off_0, unk_off_1, unk_off_2
    )

    bs.seek(file_off_1)
    logger.debug("Layer 2 offset: 0x%X", bs.tell())
    flags = bs.read_uint32()
    bs.seek(8, 1)
    unk_off_0 = bs.read_uint32() - base
    bs.seek(64, 1)
    model_wrapper_off = bs.read_uint32() - base
    unk_off_1 = bs.read_uint32() - base
    logger.debug("Unknown offsets (layer 2): 0x%X, 0x%X", unk_off_0, unk_off_1)

    bs.seek(model_wrapper_off)
    logger.debug("Model wrapper offset: 0x%X", bs.tell())
    sig = bs.read_int32()
    matrix_off = bs.read_uint32() - base
    model_off = bs.read_uint32() - base

    bs.seek(matrix_off)
    logger.debug("Matrix offset: 0x%X", bs.tell())
    unk_matrix_0 = bs.read_matrix_4x4()
    unk_matrix_1 = bs.read_matrix_4x4()
    sig = bs.read_int32()
    bs.read_int32()
    bs.seek(24, 1)
    model_0_off = bs.read_uint32() - base
    model_1_off = bs.read_uint32() - base
    bs.seek(8, 1)
    unk_off_0 = bs.read_uint32() - base
    unk_off_1 = bs.read_uint32() - base
    bs.read_int32()
    unk_off_2 = bs.read_uint32() - base
    unk_off_3 = bs.read_uint32() - base
    unk_off_4 = bs.read_uint32() - base
    unk_off_5 = bs.read_uint32() - base
    logger.debug(
        "Unknown offsets (matrix): 0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X",
        unk_off_0,
        unk_off_1,
        unk_off_2,
        unk_off_3,
        unk_off_4,
        unk_off_5,
    )

    bs.seek(model_off)
    models: list[Model] = []
    _read_model(bs, base, models)
    return models


def read_str_2(f: BufferedReader) -> list[Model]:
    bs = BinaryReader(f.read())

    bs.seek(0x1C)
    racer_type = bs.read_int32()
    if racer_type == 14:
        base = 0x8F61670
    elif racer_type == 15:
        base = 0x90C7CE0
    else:
        raise ValueError(f"Unexpected racer file type {racer_type}")

    bs.seek(0x5500)
    wrapper_2_off = bs.read_uint32() - base

    bs.seek(wrapper_2_off)
    logger.debug("Wrapper 2 offset: 0x%X", bs.tell())
    flags = bs.read_uint32()
    file_header_off = bs.read_uint32() - base
    bs.seek(92, 1)
    unk_off_0 = bs.read_uint32() - base
    bs.seek(8, 1)
    unk_off_1 = bs.read_uint32() - base
    wrapper_1_off = bs.read_uint32() - base
    logger.debug("Unknown offsets: 0x%X, 0x%X\n", unk_off_0, unk_off_1)

    bs.seek(wrapper_1_off)
    logger.debug("Wrapper 1 offset: 0x%X", bs.tell())
    flags = bs.read_uint32()
    bs.read_int32()
    bs.read_int32()
    unk_off_0 = bs.read_uint32() - base
    bs.read_vec4f()
    bs.read_vec4f()
    bs.read_vec4f()
    bs.read_vec4f()
    wrapper_0_off = bs.read_uint32() - base
    parent_off = bs.read_uint32() - base
    logger.debug("Unknown offset: 0x%X\n", unk_off_0)

    bs.seek(wrapper_0_off)
    logger.debug("Wrapper 0 offset: 0x%X", bs.tell())
    sig = bs.read_int32()
    if sig != 71:
        raise ValueError(f"Expected wrapper 0 signature 71; got {sig}")
    matrix_off = bs.read_uint32() - base
    model_off = bs.read_uint32() - base

    bs.seek(model_off)
    models: list[Model] = []
    _read_model(bs, base, models)
    return models
