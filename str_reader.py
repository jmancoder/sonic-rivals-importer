from dataclasses import dataclass
from enum import Enum
from io import BufferedReader
import logging
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from .binary_reader import BinaryReader, GECommand

logger = logging.getLogger(__name__)

RACER_BASE_0 = 0x8E149A0
RACER_BASE_1 = 0x9047CE0


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


@dataclass
class DisplayList:
    primitives: list[Primitive]
    vertices: npt.NDArray
    vertex_flags: VertexFlags


class Model(NamedTuple):
    name: str
    display_lists: list[DisplayList]


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


def _read_display_lists(bs: BinaryReader) -> list[DisplayList]:
    geometry_off = bs.tell()
    bs.read_int32()
    bs.read_int32()
    bs.read_int32()
    bs.read_int32()
    bs.read_vec4f()
    display_list_chunk_size = bs.read_uint32()
    display_list_count = bs.read_int32()

    # Read display list offsets
    display_list_offs: list[int] = []
    for _ in range(display_list_count):
        while True:
            unk_int = bs.read_int32()
            logger.debug("Skipping unknown int %i at 0x%x", unk_int, bs.tell() - 4)
            if unk_int == -1:
                break
        display_list_offs.append(bs.read_uint32())

    # Read PSP GE commands
    display_lists: list[DisplayList] = []
    for rel_off in display_list_offs:
        # Read vertex layout
        bs.seek(geometry_off + rel_off)
        command, argument = bs.read_ge_command()
        if command != GECommand.VTYPE:
            logger.error("Expected VTYPE command; got %s", command)
            continue
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
        display_list = DisplayList([], np.array([]), vertex_flags)
        list_vertex_count = 0
        while True:
            command, argument = bs.read_ge_command()
            if command == GECommand.RET:
                break
            if command != GECommand.PRIM:
                logger.warning(
                    "Unimplemented GE command %s at 0x%x", command, bs.tell() - 4
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
    logger.debug("Vertex start offset: 0x%x", bs.tell())
    for display_list in display_lists:
        display_list.vertices[...] = np.frombuffer(
            bs.getbuffer(),
            display_list.vertices.dtype,
            display_list.vertices.size,
            bs.tell(),
        )
        bs.seek(display_list.vertices.nbytes, 1)
    return display_lists


def _read_model(bs: BinaryReader, base: int) -> Model:
    sig = bs.read_uint32()
    next_off = bs.read_uint32() - base
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
    bs.read_uint32()
    bs.read_uint32()
    material_off_0 = bs.read_uint32() - base
    display_list_off = bs.read_uint32() - base
    material_off_1 = bs.read_uint32() - base
    unk_off_0 = bs.read_uint32() - base
    material_off_2 = bs.read_uint32() - base
    unk_off_1 = bs.read_uint32() - base
    material_off_3 = bs.read_uint32() - base
    unk_off_2 = bs.read_uint32() - base
    material_off_4 = bs.read_uint32() - base
    unk_off_3 = bs.read_uint32() - base
    bs.seek(name_off)
    name = bs.read_aligned_cstring()
    bs.seek(display_list_off)
    display_lists = _read_display_lists(bs)
    logger.debug(
        "Unknown model offsets: 0x%x, 0x%x, 0x%x, 0x%x",
        unk_off_0,
        unk_off_1,
        unk_off_2,
        unk_off_3,
    )
    return Model(name, display_lists)


def read_str(f: BufferedReader) -> Model:
    bs = BinaryReader(f.read())

    base = RACER_BASE_0
    bs.seek(0x7690)
    file_off_0 = bs.read_uint32() - base

    bs.seek(file_off_0)
    flags = bs.read_uint32()
    unk_off_0 = bs.read_uint32() - base
    bs.seek(80, 1)
    unk_off_1 = bs.read_uint32() - base
    bs.seek(8, 1)
    unk_off_2 = bs.read_uint32() - base
    file_off_1 = bs.read_uint32() - base
    logger.debug(
        "Unknown offsets (layer 0): 0x%x, 0x%x, 0x%x", unk_off_0, unk_off_1, unk_off_2
    )

    bs.seek(file_off_1)
    flags = bs.read_uint32()
    bs.seek(8, 1)
    unk_off_0 = bs.read_uint32() - base
    bs.seek(64, 1)
    file_off_2 = bs.read_uint32() - base
    unk_off_1 = bs.read_uint32() - base
    logger.debug("Unknown offsets (layer 1): 0x%x, 0x%x", unk_off_0, unk_off_1)

    bs.seek(file_off_2)
    sig = bs.read_uint32()
    unk_off_0 = bs.read_uint32() - base
    model_off = bs.read_uint32() - base
    logger.debug("Unknown offset (layer 2): 0x%x", unk_off_0)

    bs.seek(model_off)
    model = _read_model(bs, base)
    return model
