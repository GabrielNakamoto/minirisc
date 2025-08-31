# tiny naive assembler?

from enum import Enum
from consts import *
import string
import struct

def assemble(source):
# (1) Lexer
	lines = source.split('\n')[:-1]
	token_stream = [[] for _ in range(len(lines))]
	for k, line in enumerate(lines):
		# lex tokens
		i = 0
		while i < len(line):
			tk = line[i]

			if tk.isspace():
				pass
			elif tk == ',':
				token_stream[k].append((tk, Token.comma))
			elif tk == '(':
				token_stream[k].append((tk, Token.lparen))
			elif tk == ')':
				token_stream[k].append((tk, Token.rparen))
			elif tk.isdigit() or tk == '-':
				token = ''
				while i < len(line) and line[i] in '0123456789abcdefABCDEFxX-':
					token += line[i]
					i += 1
				token_stream[k].append((token, Token.immediate))
				continue
			elif tk=='#':  # comment
				break
			else:
				token = ''
				while i < len(line) and not line[i] in ',() \t\n':
					token += line[i]
					i += 1
				token_stream[k].append((token, Token.symbol))
				continue
			i += 1
	
	for x in token_stream:
		print(x)	

# (2) Parser

	# label pass
	symbols = []
	pc = 0x10000
	for instr in token_stream:
		token = instr[0][0]
		if token[-1] == ':': symbols.append((token[:-1], pc))
		else: pc += 4

	# instruction pass
	instructions = []
	pc = 0x10000
	for instr in token_stream:
		token = instr[0][0]
		assert instr[0][1] == Token.symbol

		if token[-1] == ':':
			continue

		pc += 4
		if token[0] == '.': # directive
			pass
		else: # op
			op = token.upper()
			enc = 0
			if op in RegOps._member_names_ + ['SUB']: # <op> rd, rs1, rs2
				rd = Regs[instr[1][0]].value
				rs1 = Regs[instr[3][0]].value
				rs2 = Regs[instr[5][0]].value
				func3 = RegOps[op].value
				func7 = 0x20 if op in ['SUB', 'SRA'] else 0x00

				enc = (func7 << 25) | (rs2 << 20)  | (rs1 << 15) | (func3 << 12) | (rd << 7) | 0b0110011
			elif op in ImmOps._member_names_ + ['SRAI']:
				rd = Regs[instr[1][0]].value
				rs1 = Regs[instr[3][0]].value
				imm = int(instr[5][0])
				func3 = ImmOps[op].value

				if op in ['SLLI', 'SRLI', 'SRAI']:
					imm &= 0x1F
					if op == "SRAI": imm |= 0x20 << 5

				enc = (imm << 20) | (rs1 << 15) | (func3 << 12) | (rd << 7) | 0b0010011
			elif op in LdOps._member_names_: # <op> rd, imm(rs1)
				rd = Regs[instr[1][0]].value
				imm = int(instr[3][0])
				rs1 = Regs[instr[5][0]].value
				func3 = LdOps[op].value

				enc = (imm << 20) | (rs1 << 15) | (func3 << 12) | (rd << 7) | 0b0000011
			elif op in StrOps._member_names_: # <op> rs2, imm(rs1)
				rs2 = 	Regs[instr[1][0]].value
				imm = int(instr[3][0])
				rs1 = Regs[instr[5][0]].value
				func3 = StrOps[op].value

				im1 = imm & 0xF
				im2 = imm >> 4

				enc = (im2 << 25) | (rs1 << 15) | (func3 << 12) | (im1 << 7) | 0b0100011
			else:
				continue
			instructions.append(enc)

	print("Symbols:", symbols)
	print("Instructions:", instructions)

	with open('output.bin', 'wb') as f:
		for x in instructions:
			f.write(struct.pack("i", x))
	

source = open('input.s', 'r').read()
assemble(source)

