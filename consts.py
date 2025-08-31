from enum import Enum


# https://www.cs.sfu.ca/~ashriram/Courses/CS295/assets/notebooks/RISCV/RISCV_CARD.pdf

# ----- Operations -----
class RegOps(Enum):
	ADD = SUB = 0x0
	XOR = 0x4
	OR = 0x6
	AND = 0x7
	SLL = 0x1
	SRL = SRA = 0x5
	SLT = 0x2
	SLTU = 0x3

class ImmOps(Enum):
	ADDI = 0x0
	XORI = 0x4
	ORI = 0x6
	ANDI = 0x7
	SLLI = 0x1
	SRLI = SRAI = 0x5
	SLTI = 0x2
	SLTIU = 0x3

class LdOps(Enum):
	LB = 0x0
	LH = 0x1
	LW = 0x2
	LBU = 0x4
	LHU = 0x5

class StrOps(Enum):
	SB = 0x0
	SH = 0x1
	SW = 0x2

class BrchOps(Enum):
	BEQ = 0x0
	BNE = 0x1
	BLT = 0x4
	BGE = 0x5
	BLTU = 0x6
	BGEU = 0x7
	
class JmpOps(Enum):
	JAL = 0xF # value doesnt matter
	JALR = 0x0

class EnvOps(Enum):
	ECALL = EBREAK = 0x0

class UimmOps(Enum): # values dont matter
	LUI = 0x0
	AUIPC = 0x1
	
# ----- Token types -----
class Token(Enum):
	symbol = 9
	comma = 1
	immediate = 2
	lparen = 3
	rparen = 4

# ----- Register mnemonics ------
class Regs(Enum):
	x0 = zero = 0
	x1 = ra = 1
	x2 = sp = 2
	x3 = gp = 3
	x4 = tp = 4
	x5 = t0 = 5
	x6 = t1 = 6
	x7 = t2 = 7
	x8 = s0 = fp = 8
	x9 = s1 = 9
	x10 = a0 = 10
	x11 = a1 = 11
	x12 = a2 = 12
	x13 = a3 = 13
	x14 = a4 = 14
	x15 = a5 = 15
	x16 = a6 = 16
	x17 = a7 = 17
	x18 = s2 = 18
	x19 = s3 = 19
	x20 = s4 = 20
	x21 = s5 = 21
	x22 = s6 = 22
	x23 = s7 = 23
	x24 = s8 = 24
	x25 = s9 = 25
	x26 = s10 = 26
	x27 = s11 = 27
	x28 = t3 = 28
	x29 = t4 = 29
	x30 = t5 = 30
	x31 = t6 = 31

