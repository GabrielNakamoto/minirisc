# Mini RISC-V 32  Assembler, 2025 <gabriel@nakamoto.ca>

from enum import Enum
from consts import *
import string
import struct
import sys
import os

def debug(*args, **kwargs):
	if os.environ.get("RIVER_DEBUG"): print(*args, **kwargs)

def build_err(e, instr, state):
	badtk = str(e).split(':')[1].strip()
	descr = f"\033[1m{state.infile}:{state.ln}\033[0m :: '{e}'\n|\t"
	srcline = " ".join(f"\033[1;31m{tk[0]}\033[0m" if tk[0] == badtk else str(tk[0]) for tk in instr)
	state.errs.append(descr+srcline)
	state.failed = 1

def expand_rrx(op, r1, r2, i, symbol=False):
	return [
		(op, Token.symbol),
		(r1, Token.symbol),
		(',', Token.comma),
		(r2, Token.symbol),
		(',', Token.comma),
		(i, Token.immediate if not symbol else Token.symbol)
	]

def expand_ri(op, r1, i):
	return [
		(op, Token.symbol),
		(r1, Token.symbol),
		(',', Token.comma),
		(i, Token.immediate)
	]

def expand_rrsi(op, rd, rs, i):
	return [
		(op, Token.symbol),
		(rd, Token.symbol),
		(',', Token.comma),
		(i, Token.immediate),
		('(', Token.lparen),
		(rs, Token.symbol),
		(')', Token.rparen)
	]

def split_immediate(imm):
	imm31_12 = (imm >> 12) & 0xFFFFF
	imm11_0 = imm & 0xFFF
	msb = imm11_0 >> 11
	# handle sign extension
	if msb: imm31_12 = (imm31_12 + 1) & 0xFFFFF 

	return (imm31_12, imm11_0)

def branch_zero(x, state): # <op> rs1, rs2, imm
	rs1 = x[1][0]
	imm = x[3][0]
	base_op = x[0][0][:-1].lower()

	return [expand_rrx(base_op, rs1, 'x0', imm, imm in state.symbols)]

def li(x, state): # naive expansion?
	rd = x[1]
	imm = parse_imm(x[3])
	imm31_12, imm11_0 = split_immediate(imm)

	addi = expand_rrx("addi", rd[0], 'x0', hex(imm11_0)) 
	lui = expand_ri("lui", rd[0], hex(imm31_12))

	# return seq
	return [lui, addi]

def la(x, state):
	rd = x[1][0]
	addr = state.symbols[x[3][0]]
	addr_h, addr_l = split_immediate(addr)

	addi = expand_rrx("addi", rd, rd, hex(addr_l))
	auipc = expand_ri("auipc", rd, hex(addr_h))

	return [auipc, addi]

def nop(x, state):
	return [expand_rrx("addi", 'x0', 'x0', '0x0')]

def mv(x, state):
	rd = x[1][0]
	rs = x[3][0]
	return [expand_rrx("addi", rd, rs, '0x0')]

def neg(x, state):
	rd = x[1][0]
	rs = x[3][0]
	return [expand_rrx("sub", rd, 'x0', rs)]

def call(x, state):
	# if < 12 bit offset only use jal
	offset = parse_pc_offs(x[1], state)
	offs_h, offs_l = split_immediate(offset)
	auipc = expand_ri("auipc", 'x1', hex(offs_h)) # store upper part of target address in x1
	jalr = expand_rrsi("jalr", 'x1', 'x1', hex(offs_l)) # jump to upper part + lower part of address

	return [auipc, jalr]

def ret(x, state):
	return [expand_rrsi("jalr", 'x0', 'x1', '0')]

pseud_map = [ # (worst case # of ops, expansion func)
	(1, branch_zero),
	(2, li),
	(2, la),
	(1, mv),
	(1, nop),
	(1, neg),
	(2, call),
	(1, ret)
]

def parse_imm(x):
	try:
		return int(x[0], 16) if len(x[0]) > 1 and x[0][1].lower() == 'x' else int(x[0])
	except Exception as e:
		raise Exception(f"Error parsing immediate: {x[0]}")

def parse_rsi(x): 
	# Common: rs, imm | imm(rs)
	# Only allow canonical: imm(rs)
	imm = parse_imm(x[0])
	rs = parse_reg(x[2])
	return (rs, imm)

def parse_reg(x):
	try:
		return Regs[x[0]].value
	except Exception as e:
		raise Exception(f"Error parsing register: {x[0]}")

def parse_pc_offs(x, state):
	if x[1] == Token.symbol:
		try:
			return state.symbols[x[0]] - state.pc
		except Exception as e:
			raise Exception(f"Error calculating symbol offset: {x[0]}")
	else:
		parse_imm(x)

def encode_i_type(imm, rs1, func3, rd, opc):
	return (imm << 20) | (rs1 << 15) | (func3 << 12) | (rd << 7) | opc

def encode_r_type(func7, rs2, rs1, func3, rd, opc):
	return (func7 << 25) | (rs2 << 20) | (rs1 << 15) | (func3 << 12) | (rd << 7) | opc

def encode_s_type(imm, rs2, rs1, func3, opc):
	return ((imm >> 4) << 25) | (rs2 << 20) | (rs1 << 15) | (func3 << 12) | ((imm & 0x1F) << 7) | opc

def encode_j_type(imm, rd, opc):
	# lsb not stored, all shifts > by 1
	# imm[20|10:1|11|19:12]
	im20 = (imm >> 20) & 0x1
	im10_1 = (imm >> 1) & 0x3FF
	im11 = (imm >> 11) & 0x1
	im19_12 = (imm >> 12) & 0xFF

	return (im20 << 31) | (im10_1 << 21) | (im11 << 20) | (im19_12 << 12) | (rd << 7) | opc

def encode_b_type(imm, rs2, rs1, func3, opc):
	# lsb not stored, all shifts > by 1
	im12 = (imm >> 12) & 0x1
	im10_5 = (imm >> 5) & 0x3f
	im11 = (imm >> 11) & 0x1
	im4_1 = (imm >> 1) & 0xf

	imh = (im12 << 6) | im10_5 # imm[12|10:5]
	iml = (im4_1 << 1) | im11 # imm[4:1|11]

	return (imh << 25) | (rs2 << 20) | (rs1 << 15) | (func3 << 12) | (iml << 7) | opc

def encode_u_type(imm, rd, opc):
	return (imm << 12) | (rd << 7) | opc

class AssemblerState:
	start_addr = 0x0

	def __init__(self, infile, outfile="output.bin"):
		self.pc = AssemblerState.start_addr
		self.symbols = dict()
		self.infile = infile
		self.outfile = outfile
		self.failed = 0 
		self.line_nmap = []
		self.ln = 0
		self.errs = []

	def reset_pc(self):
		self.pc = AssemblerState.start_addr

def lex(source): # => token stream
	lines = source.split('\n')[:-1]
	token_stream = []
	for k, line in enumerate(lines):
		# lex tokens
		i = 0
		tokens = []
		while i < len(line):
			tk = line[i]

			if tk.isspace():
				pass
			elif tk == ',':
				tokens.append((tk, Token.comma))
			elif tk == '(':
				tokens.append((tk, Token.lparen))
			elif tk == ')':
				tokens.append((tk, Token.rparen))
			elif tk.isdigit() or tk == '-':
				token = ''
				while i < len(line) and line[i] in '0123456789abcdefABCDEFxX-':
					token += line[i]
					i += 1
				tokens.append((token, Token.immediate))
				continue
			elif tk=='#':  # comment
				break
			else:
				token = ''
				while i < len(line) and not line[i] in ',() \t\n':
					token += line[i]
					i += 1
				tokens.append((token, Token.symbol))
				continue
			i += 1
		if len(tokens) > 0 and tokens[0][0] != '#': token_stream.append((k+1, tokens)) # (source line #, tokens)
	
	debug("Tokens:")
	for i, x in enumerate(token_stream):
		debug(f'\t{i}:', end='\t')
		for tk in x[1]:
			debug(tk[0], tk[1].name, end='\t')
		debug()

	return token_stream

def label_pass(token_stream, state):
	state.reset_pc()
	for i, instr in enumerate(token_stream):
		token = instr[0][0]
		if token[0] == '.': continue
		if token[-1] == ':':
			state.symbols[token[:-1]] = state.pc
		else: state.pc += 4

def resolve_pass(token_stream, state):
	state.reset_pc()
	debug("Resolve pass:")
	for i, instr in enumerate(token_stream):
		token = instr[0][0]
		if token[0] == '.' or token[-1] == ':': continue
		op = token.lower()
		debug('\t<' + hex(state.pc) + '>', '\t', token)
		if op in list(PseudOps.__members__.keys()):
			exp_len = pseud_map[PseudOps[op].value][0]
			delta = (exp_len - 1) * 4
			for symbol, addr in state.symbols.items():
				if addr > state.pc:
					debug("\t\tResolving:", symbol, f'{hex(addr)} -> {hex(addr+delta-4)}')
					state.symbols[symbol] = addr+delta
		state.pc += 4

	debug("Symbols:")
	for s, x in state.symbols.items():
		debug('\t', s, f'<{hex(x)}>')

def expansion_pass(token_stream, state):
	state.reset_pc()
	debug("Expansion pass:")
	for i, line in enumerate(token_stream):
		token = line[0][0]
		if token[0] == '.' or token[-1] == ':': continue
		if token.lower() not in list(PseudOps.__members__.keys()):
			debug('\t<' + hex(state.pc) + '>', '\t', token)
		else:
			token_stream.pop(i)
			ln = state.line_nmap.pop(i)
			state.ln = ln
			idx = PseudOps[token.lower()].value
			try:
				expansion = pseud_map[idx][1](line, state)
			except Exception as e:
				build_err(e, line, state)
			for j, x in enumerate(expansion):
				token_stream.insert(i+j, x)
				state.line_nmap.insert(i+j, ln)

			debug(f'\t<{hex(state.pc)}>', '\t', token, '=>', [x[0][0] for x in expansion])
		state.pc += 4

def encode(token_stream, state, output='output.bin'):
	instructions = []

	state.reset_pc()
	debug("Encoding pass:")
	for k, instr in enumerate(token_stream):
		state.ln = state.line_nmap[k]
		token = instr[0][0]
		assert instr[0][1] == Token.symbol

		if token[-1] == ':':
			# TODO: handle inline labels ex. foo: add t1, zero, t4
			continue
		if token[0] == '.':
			pass
		else: # op
			debug("\t", ' '.join([str(tk[0]) for tk in instr]))
			op = token.upper()
			enc = 0
			try:
				if op in RegOps._member_names_ + ['SUB'] + MulOps._member_names_: # <op> rd, rs1, rs2
					rd = parse_reg(instr[1])
					rs1 = parse_reg(instr[3])
					rs2 = parse_reg(instr[5])
					if op in MulOps._member_names_:
						func3 = MulOps[op].value
						func7 = 0x01
					else:
						func3 = RegOps[op].value
						func7 = 0x20 if op in ['SUB', 'SRA'] else 0x00
					enc = encode_r_type(func7, rs2, rs1, func3, rd, 0b0110011)
				elif op in ImmOps._member_names_ + ['SRAI']:
					rd = parse_reg(instr[1])
					rs1 = parse_reg(instr[3])
					imm = parse_imm(instr[5])
					func3 = ImmOps[op].value

					if op in ['SLLI', 'SRLI', 'SRAI']:
						imm &= 0x1F
						if op == "SRAI": imm |= 0x20 << 5

					enc = encode_i_type(imm, rs1, func3, rd, 0b0010011)
				elif op in LdOps._member_names_: # <op> rd, imm(rs1)
					rd = parse_reg(instr[1])
					rs1, imm = parse_rsi(instr[3:])
					func3 = LdOps[op].value

					enc = encode_i_type(imm, rs1, func3, rd, 0b0000011)
				elif op in StrOps._member_names_: # <op> rs2, imm(rs1)
					rs2 = parse_reg(instr[1])
					rs1, imm = parse_rsi(instr[3:])
					func3 = StrOps[op].value

					enc = encode_s_type(imm, rs2, rs1, func3, 0b0100011)
				elif op in EnvOps._member_names_ + ['EBREAK']:
					imm = 0x0 if op == 'ECALL' else 0x1
					enc = encode_i_type(imm, 0, 0x0, 0, 0b1110011)
				elif op in BrchOps._member_names_: # <op> rs1, rs2, <imm|label>
					rs1 = parse_reg(instr[1])
					rs2 = parse_reg(instr[3])
					imm = parse_pc_offs(instr[5], state)
					func3 = BrchOps[op].value

					enc = encode_b_type(imm, rs2, rs1, func3, 0b1100011)
				elif op in JmpOps._member_names_:
					rd = parse_reg(instr[1])
					if op == 'JALR':
						rs, imm = parse_rsi(instr[3:])
						enc = encode_i_type(imm, rs, 0x0, rd, 0b1100111)
					else: 
						imm = parse_pc_offs(instr[3], state)
						enc = encode_j_type(imm, rd, 0b1101111)
				elif op in UimmOps._member_names_:
					rd = parse_reg(instr[1])
					
					imm = parse_imm(instr[3]) if instr[3][1] == Token.immediate else (parse_pc_offs(instr[3], state) >> 12)
					opc = 0b0110111	 if op == 'LUI' else 0b0010111
					enc = encode_u_type(imm, rd, opc)
				else:
					continue
			except Exception as e:
				build_err(e, instr, state)
			instructions.append((op, enc))
			state.pc += 4

	if state.failed:
		for err in state.errs:
			print(err)
		os._exit(1)	

	debug("Instructions:")
	for i in instructions:
		debug('\t', f'{i[0]}\t\t', bin(i[1]).split('0b')[1].zfill(32))

	with open(state.outfile, 'wb') as f:
		for x in instructions:
			f.write(struct.pack("I", x[1]))

	print(f"\033[1mriver ::\033[0m Succesfully assembled machine code -> \033[1;32m{state.outfile}\033[0m")

def assemble(source, filename):
	debug("Source:\n", source)

	state = AssemblerState(filename)

	token_stream = lex(source)

	for i, x in enumerate(token_stream):
		state.line_nmap.append(x[0])
		token_stream[i] = x[1]

	label_pass(token_stream, state)
	resolve_pass(token_stream, state)
	expansion_pass(token_stream, state)

	encode(token_stream, state)

def main():
	source = open(sys.argv[1], 'r').read()
	assemble(source, sys.argv[1])

if __name__	== "__main__":
	main()
