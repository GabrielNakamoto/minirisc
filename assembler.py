# tiny naive assembler?

from enum import Enum
from consts import *
import string

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
				i += 1
			elif tk == ',':
				token_stream[k].append((tk, Token.comma))
				i += 1
			elif tk.isdigit() or tk == '-':
				token = ''
				while i < len(line) and line[i] in '0123456789abcdefABCDEFxX-':
					token += line[i]
					i += 1
				token_stream[k].append((token, Token.immediate))
			elif tk=='#':  # comment
				break
			else:
				token = ''
				while i < len(line) and (not line[i].isspace()) and line[i] != ',':
					token += line[i]
					i += 1
				token_stream[k].append((token, Token.symbol))
	
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
			if op in RegOps._member_names_: # <op> rd, rs1, rs2
				rd = Regs[instr[1][0]].value
				rs1 = Regs[instr[3][0]].value
				rs2 = Regs[instr[5][0]].value
				func3 = RegOps[op].value
				func7 = 0x20 if op in ['SUB', 'SRA'] else 0x00

				enc = (func7 << 25) | (rs2 << 20)  | (rs1 << 15) | (func3 << 12) | (rd << 7) | 0b0110011
				instructions.append(bin(enc))

	print("Symbols:", symbols)
	print("Instructions:", instructions)

source = open('input.s', 'r').read()
assemble(source)

