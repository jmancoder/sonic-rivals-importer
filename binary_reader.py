from enum import Enum
from io import BytesIO, StringIO
import struct
from typing import Literal

from mathutils import Matrix


class GECommand(Enum):
    NOP = 0x00
    VADR = 0x01
    IADR = 0x02
    PRIM = 0x04
    BEZIER = 0x05
    SPLINE = 0x06
    BBOX = 0x07
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
    LTE = 0x17
    LE0 = 0x18
    LE1 = 0x19
    LE2 = 0x1A
    LE3 = 0x1B
    CLE = 0x1C
    BCE = 0x1D
    TME = 0x1E
    FGE = 0x1F
    DTE = 0x20
    ABE = 0x21
    ATE = 0x22
    ZTE = 0x23
    STE = 0x24
    AAE = 0x25
    PCE = 0x26
    CTE = 0x27
    LOE = 0x28
    BONEN = 0x2A
    BONED = 0x2B
    WEIGHT0 = 0x2C
    WEIGHT1 = 0x2D
    WEIGHT2 = 0x2E
    WEIGHT3 = 0x2F
    WEIGHT4 = 0x30
    WEIGHT5 = 0x31
    WEIGHT6 = 0x32
    WEIGHT7 = 0x33
    DIVIDE = 0x36
    PPM = 0x37
    PFACE = 0x38
    WORLDN = 0x3A
    WORLDD = 0x3B
    VIEWN = 0x3C
    VIEWD = 0x3D
    PROJN = 0x3E
    PROJD = 0x3F
    TGENN = 0x40
    TGEND = 0x41
    SX = 0x42
    SY = 0x43
    SZ = 0x44
    TX = 0x45
    TY = 0x46
    TZ = 0x47
    SU = 0x48
    SV = 0x49
    TU = 0x4A
    TV = 0x4B
    OFFSETX = 0x4C
    OFFSETY = 0x4D
    SHADE = 0x50
    NREV = 0x51
    MATERIAL = 0x53
    MEC = 0x54
    MAC = 0x55
    MDC = 0x56
    MSC = 0x57
    MAA = 0x58
    MK = 0x5B
    AC = 0x5C
    AA = 0x5D
    LMODE = 0x5E
    LTYPE0 = 0x5F
    LTYPE1 = 0x60
    LTYPE2 = 0x61
    LTYPE3 = 0x62
    LX0 = 0x63
    LY0 = 0x64
    LZ0 = 0x65
    LX1 = 0x66
    LY1 = 0x67
    LZ1 = 0x68
    LX2 = 0x69
    LY2 = 0x6A
    LZ2 = 0x6B
    LX3 = 0x6C
    LY3 = 0x6D
    LZ3 = 0x6E
    LDX0 = 0x6F
    LDY0 = 0x70
    LDZ0 = 0x71
    LDX1 = 0x72
    LDY1 = 0x73
    LDZ1 = 0x74
    LDX2 = 0x75
    LDY2 = 0x76
    LDZ2 = 0x77
    LDX3 = 0x78
    LDY3 = 0x79
    LDZ3 = 0x7A
    LKA0 = 0x7B
    LKB0 = 0x7C
    LKC0 = 0x7D
    LKA1 = 0x7E
    LKB1 = 0x7F
    LKC1 = 0x80
    LKA2 = 0x81
    LKB2 = 0x82
    LKC2 = 0x83
    LKA3 = 0x84
    LKB3 = 0x85
    LKC3 = 0x86
    LKS0 = 0x87
    LKS1 = 0x88
    LKS2 = 0x89
    LKS3 = 0x8A
    LKO0 = 0x8B
    LKO1 = 0x8C
    LKO2 = 0x8D
    LKO3 = 0x8E
    LAC0 = 0x8F
    LDC0 = 0x90
    LSC0 = 0x91
    LAC1 = 0x92
    LDC1 = 0x93
    LSC1 = 0x94
    LAC2 = 0x95
    LDC2 = 0x96
    LSC2 = 0x97
    LAC3 = 0x98
    LDC3 = 0x99
    LSC3 = 0x9A
    CULL = 0x9B
    FBP = 0x9C
    FBW = 0x9D
    ZBP = 0x9E
    ZBW = 0x9F
    TBP0 = 0xA0
    TBP1 = 0xA1
    TBP2 = 0xA2
    TBP3 = 0xA3
    TBP4 = 0xA4
    TBP5 = 0xA5
    TBP6 = 0xA6
    TBP7 = 0xA7
    TBW0 = 0xA8
    TBW1 = 0xA9
    TBW2 = 0xAA
    TBW3 = 0xAB
    TBW4 = 0xAC
    TBW5 = 0xAD
    TBW6 = 0xAE
    TBW7 = 0xAF
    CBP = 0xB0
    CBW = 0xB1
    XBP1 = 0xB2
    XBW1 = 0xB3
    XBP2 = 0xB4
    XBW2 = 0xB5
    TSIZE0 = 0xB8
    TSIZE1 = 0xB9
    TSIZE2 = 0xBA
    TSIZE3 = 0xBB
    TSIZE4 = 0xBC
    TSIZE5 = 0xBD
    TSIZE6 = 0xBE
    TSIZE7 = 0xBF
    TMAP = 0xC0
    TSHADE = 0xC1
    TMODE = 0xC2
    TPF = 0xC3
    CLOAD = 0xC4
    CLUT = 0xC5
    TFILTER = 0xC6
    TWRAP = 0xC7
    TLEVEL = 0xC8
    TFUNC = 0xC9
    TEC = 0xCA
    TFLUSH = 0xCB
    TSYNC = 0xCC
    FOG1 = 0xCD
    FOG2 = 0xCE
    FC = 0xCF
    TSLOPE = 0xD0
    FPF = 0xD2
    CMODE = 0xD3
    SCISSOR1 = 0xD4
    SCISSOR2 = 0xD5
    MINZ = 0xD6
    MAXZ = 0xD7
    CTEST = 0xD8
    CREF = 0xD9
    CMSK = 0xDA
    ATEST = 0xDB
    STEST = 0xDC
    SOP = 0xDD
    ZTEST = 0xDE
    BLEND = 0xDF
    FIXA = 0xE0
    FIXB = 0xE1
    DITH1 = 0xE2
    DITH2 = 0xE3
    DITH3 = 0xE4
    DITH4 = 0xE5
    LOP = 0xE6
    ZMSK = 0xE7
    PMSK1 = 0xE8
    PMSK2 = 0xE9
    XSTART = 0xEA
    XPOS1 = 0xEB
    XPOS2 = 0xEC
    XSIZE = 0xEE


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

    def read_matrix_4x4(self) -> Matrix:
        return Matrix([self.read_vec4f() for _ in range(4)]).transposed()

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
