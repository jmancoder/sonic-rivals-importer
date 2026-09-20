from enum import Enum
from io import BytesIO, StringIO
import struct
from typing import Literal


class GECommand(Enum):
    NOP = 0x00
    VADDR = 0x01
    IADDR = 0x02
    PRIM = 0x04
    BEZIER = 0x05
    SPLINE = 0x06
    BOUNDINGBOX = 0x07
    JUMP = 0x08
    BJUMP = 0x09
    CALL = 0x0A
    RET = 0x0B
    END = 0x0C
    SIGNAL = 0x0E
    FINISH = 0x0F
    BASE = 0x10
    VTYPE = 0x12
    OFFSET = 0x13
    ORIGIN = 0x14
    REGION1 = 0x15
    REGION2 = 0x16
    LIGHTING = 0x17
    LIGHT0 = 0x18
    LIGHT1 = 0x19
    LIGHT2 = 0x1A
    LIGHT3 = 0x1B
    DEPTH_CLIP = 0x1C
    CULL_FACE = 0x1D
    TEXTURE = 0x1E
    FOG = 0x1F
    DITHER = 0x20
    ALPHA_BLEND = 0x21
    ALPHA_TEST = 0x22
    Z_TEST = 0x23
    STENCIL = 0x24
    ANTIALIAS = 0x25
    PATCH_CULL = 0x26
    COLOR_TEST = 0x27
    LOGIC_OP = 0x28
    BONE_NUM = 0x2A
    BONE_DATA = 0x2B
    MORPH0 = 0x2C
    MORPH1 = 0x2D
    MORPH2 = 0x2E
    MORPH3 = 0x2F
    MORPH4 = 0x30
    MORPH5 = 0x31
    MORPH6 = 0x32
    MORPH7 = 0x33
    PATCH_DIV = 0x36
    PATCH_PRIM = 0x37
    PATCH_FACING = 0x38


class BinaryReader(BytesIO):
    def __init__(self, data: bytes, big_endian=False) -> None:
        super().__init__(data)

        self.endian_symbol: Literal["<", ">"]
        self.byte_order: Literal["little", "big"]
        if big_endian:
            self.endian_symbol = ">"
            self.byte_order = "big"
        else:
            self.endian_symbol = "<"
            self.byte_order = "little"

    def read_uint8(self) -> int:
        return int.from_bytes(self.read(1), signed=False, byteorder=self.byte_order)

    def read_int8(self) -> int:
        return int.from_bytes(self.read(1), signed=True, byteorder=self.byte_order)

    def read_uint16(self) -> int:
        return int.from_bytes(self.read(2), signed=False, byteorder=self.byte_order)

    def read_int16(self) -> int:
        return int.from_bytes(self.read(2), signed=True, byteorder=self.byte_order)

    def read_uint32(self) -> int:
        return int.from_bytes(self.read(4), signed=False, byteorder=self.byte_order)

    def read_int32(self) -> int:
        return int.from_bytes(self.read(4), signed=True, byteorder=self.byte_order)

    def read_vec4B(self) -> tuple[int, int, int, int]:
        return struct.unpack(self.endian_symbol + "4B", self.read(4))

    def read_vec3H(self) -> tuple[int, int, int]:
        return struct.unpack(self.endian_symbol + "3H", self.read(6))

    def read_vec3I(self) -> tuple[int, int, int]:
        return struct.unpack(self.endian_symbol + "3I", self.read(12))

    def read_float(self) -> float:
        return struct.unpack(self.endian_symbol + "f", self.read(4))[0]

    def read_vec2f(self) -> tuple[float, float]:
        return struct.unpack(self.endian_symbol + "2f", self.read(8))

    def read_vec3f(self) -> tuple[float, float, float]:
        return struct.unpack(self.endian_symbol + "3f", self.read(12))

    def read_vec4f(self) -> tuple[float, float, float, float]:
        return struct.unpack(self.endian_symbol + "4f", self.read(16))

    def read_ge_command(self) -> tuple[GECommand, int]:
        argument = int.from_bytes(self.read(3), signed=False, byteorder=self.byte_order)
        command = GECommand(self.read_uint8())
        return command, argument

    def read_string_block(self, length: int) -> str:
        text = self.read(length).decode(errors="ignore")
        return text.split("\x00")[0]

    def read_cstring(self) -> str:
        string = StringIO()
        while True:
            char = self.read(1)
            if char == b"\x00":
                break
            string.write(char.decode())
        return string.getvalue()

    def read_aligned_cstring(self) -> str:
        text = self.read_cstring()
        self.seek(((len(text) + 3) // 4 * 4) - len(text))
        return text
