"""
ARM assembler :: <gabriel@nakamoto.ca> 2025

References:
	* https://pages.cs.wisc.edu/~markhill/restricted/arm_isa_quick_reference.pdf
	* https://developer.arm.com/documentation/ddi0487/lb/?lang=en
	* https://student.cs.uwaterloo.ca/~cs241/slides/sylvie/Sylvie-L5.pdf

Compilation = Analysis + Synthesis
"""
from enum import Enum


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

source = """MOV R4, #0      ; Initialize sum (R4) to 0
loop:
    LDR R2, [R0]    ; Load the integer at address R0 into R2
    ADD R4, R4, R2  ; Add the loaded integer (R2) to the sum (R4)
    ADD R0, R0, #4  ; Move R0 to the address of the next integer (add 4 bytes)
    SUB R1, R1, #1  ; Decrement the count of remaining integers (R1)
    CMP R1, #0      ; Compare R1 with 0
    BNE loop        ; If R1 is not equal to 0, branch back to 'loop'"""

dfa = DFA()
dfa.tokenize(source)
print(dfa)
