"""
ARM assembler :: <gabriel@nakamoto.ca> 2025

References:
	* https://pages.cs.wisc.edu/~markhill/restricted/arm_isa_quick_reference.pdf
	* https://developer.arm.com/documentation/ddi0487/lb/?lang=en
	* https://student.cs.uwaterloo.ca/~cs241/slides/sylvie/Sylvie-L5.pdf

Compilation = Analysis + Synthesis
"""
from enum import Enum
import os
from simpleelf import elf_consts
from simpleelf.elf_builder import ElfBuilder

class State(Enum):
	NONE = 0
	IDENTIFIER = 2
	DOT_IDENTIFIER = 1
	REGISTER = 3
	IMMEDIATE = 4
	RCASE = 5
	NEWLINE = 6
	COMMA = 7
	COMMENT = 8
	LPAREN = 9
	RPAREN = 10
	TAB = 11

# Deterministic Finite Automata (DFA) Tokenizer
class DFA:
	def __init__(self):
		ns = [dict() for _ in range(len(State))]

		ns[0][ord('.')] = State.DOT_IDENTIFIER 			
		ns[0][ord('#')] = State.IMMEDIATE 			
		ns[0][ord('\n')] = State.NEWLINE
		ns[0][ord(',')] = State.COMMA
		ns[0][ord(';')] = State.COMMENT
		ns[0][ord('[')] = State.LPAREN
		ns[0][ord(']')] = State.RPAREN

		for i in range(256):
			if i == ord('\n'): continue
			ns[8][i] = State.COMMENT
		for i in range(9):
			ns[5][48+i] = State.REGISTER				# Rcase -- 0-9 -> Register
			ns[4][48+i] = State.IMMEDIATE				# Immediate -- 0-9 -> Immediate
		for i in range(26):
			ns[0][ord('a')+i] = State.IDENTIFIER  		# None -- a-z -> Identif
			ns[1][ord('a')+i] = State.DOT_IDENTIFIER 	# . identif -- a-z -> . identif
			ns[2][ord('a')+i] = State.IDENTIFIER		# Identif -- a-z -> Identif
			ns[5][ord('a')+i] = State.IDENTIFIER 		# Rcase -- a-Z -> Identif

		ns[2][ord(':')] = State.IDENTIFIER				# Identif -- : -> Identif
		ns[0][ord('r')] = State.RCASE					# None -- R -> Rcase

		self.nextstate = ns
		self.ls = State.NONE
		self.state = State.NONE
		self.token = ''
		self.tokens = []
		self.log = 0

	def emit(self, token=None):
		if token == None:
			token = (self.token, self.ls.name)
		if self.log: print("Emitting:", token)
		if self.token != '': self.tokens.append(token)
		self.token = ''
		self.state = State.NONE

	def tokenize(self, source, log=0):
		self.log = log
		i = 0
		while i < len(source):
			c = source[i]

			self.ls = self.state
			self.state = self.nextstate[self.state.value].get(ord(c.lower()))

			if self.log: print(i, repr(c), self.state)

			if c == ' ' and self.ls != State.COMMENT: 
				self.emit()
			elif self.state == None:
				self.emit()
				i -= 1
			else:
				self.token += c

			i += 1

		self.emit()

	def __str__(self):
		rep = "\nToken stream:\n"
		rep += "--------------\n"
		for tk in self.tokens:
			rep += str(tk) + "\n"
		return rep


class Cond(Enum):
	EQ = 0
	NE = 1
	CS = 2
	CC = 3
	MI = 4
	PL = 5
	VS = 6
	VC = 7
	HI = 8
	LS = 9
	GE = 10
	LT = 11
	GT = 12
	LE = 13
	AL = 14

class DataOps(Enum): # data processing
	AND = 0
	EOR = 1
	SUB = 2
	RSB = 3
	ADD = 4
	ADC = 5
	SBC = 6
	RSC = 7
	TST = 8
	TEQ = 9
	CMP = 10
	CMN = 11
	ORR = 12
	MOV = 13
	BIC = 14
	MVN = 15

class MemSOps(Enum): 	# load and store (single)
	LDR = 0
	STR = 1

class BrnchOps(Enum): 	# branch + branch w link
	B = 0
	BL = 1

class Parser:
	op_types = [DataOps, MemSOps, BrnchOps]
	def __init__(self, tokens):
		self.tokens = tokens
		self.instructions = []

	def get_line(self, tokens):
		line = []
		for i, tk in enumerate(tokens):
			if tk[1] == "NEWLINE" or i == len(tokens)-1: return (line, i+1)
			if tk[1] == "COMMENT": continue
			line.append(tk)

	def parse_op2(self, op2):
		# https://developer.arm.com/documentation/den0013/0400/ARM-Thumb-Unified-Assembly-Language-Instructions/Data-processing-operations/Operand-2-and-the-barrel-shifter
		"""
		cases:
			
			ADD r0, r1, r2	; register
			ADD r0, r1, r2, LSL, #3 ; register with shift
			ADD r0, r1, #10 ; immediate
		"""

		i = 0
		bits = 0
		token = op2[0][0]
		if token[0] == 'R':
			if len(op2) == 1: # reg case
				bits = int(token[1:]) & 0xF
			else: # shift case
				bits = 0
		elif token[0] == '#':
			# immediate case
			# TODO: check if rotation required
			bits = self.parse_reg(token)
			i = 1
		return (bits, i)

	def parse_reg(self, tk):
		# TODO: check if rotation required
		bits = int(tk[1:]) & 0xFF
		return bits

	def parse(self):
		i = 0

		while i < len(self.tokens):
			line, j = self.get_line(self.tokens[i:])
			i += j
			
			token = line[0][0]

			if token[-1] == ':':
				print("Label")
			else:
				print("Instruction")

				cond = Cond.AL
				s = 0

				if token[-1] == 'S':
					s = 1
					token=token[:-1]
				if token[-2:] in Cond._member_names_:
					print("Condition!")

					cond = Cond[token[-2:]]
					print(token[-2:], bin(cond.value))
					token = token[:-2]

				if token in DataOps._member_names_:
					rn = 0
					rd = 0
					imm = 0
					opc = DataOps[token].value

					print(bin(opc))
					if token in ["CMP", "CMN", "TEQ", "TST"]: # no result, <opcode>{cond} Rn, <Op2>
						rn = self.parse_reg(line[1][0])
						op2, imm = self.parse_op2(line[3:])

					elif token in ["MOV", "MVN"]: # single operand, <opcode>{cond}{S} Rd, <Op2>
						rd = self.parse_reg(line[1][0])
						op2, imm = self.parse_op2(line[3:])

					else: # <opcode>{cond}{S} Rd, Rn, <op2>
						rd = self.parse_reg(line[1][0])
						rn = self.parse_reg(line[3][0])
						op2, imm = self.parse_op2(line[5:])

					word = (cond.value << 28) | (imm << 25) | (opc << 21) | (s << 20) | (rn << 16) | (rd << 12) | op2

					self.instructions.append(word)

				"""
				for opt in Parser.op_types:
					if token in opt._member_names_:
						print(opt.__name__)
				"""

			# cases: directive, instruction, 
			print(line)

	def __str__(self):
		ret = ""
		for w in self.instructions:
			ret += '\n' + bin(w)
		return ret

	def save(self, filename="output.elf"):
		e = ElfBuilder(elf_consts.ELFCLASS32)
		e.set_endianity('<')
		e.set_machine(elf_consts.EM_ARM)

		code = b''.join(w.to_bytes(4, byteorder='little', signed=False) for w in self.instructions)

		print("Code len", len(code))

		loadaddr = 0x10000
		e.add_segment(loadaddr, code, elf_consts.PF_R | elf_consts.PF_X)
		e.add_code_section(loadaddr, len(code), name='.text')
		e.set_entry(loadaddr)

		with open(filename, 'wb') as obj:
			obj.write(e.build())

source = open("input.s", "r").read()

dfa = DFA()
dfa.tokenize(source)
print(dfa)

parser = Parser(dfa.tokens)
parser.parse()

print(parser)

parser.save()

