<<<<<<< HEAD
# 9/25 HW:


import math
import operator as op
from typing import Any

Symbol = str
Number = (int, float)
Atom = (Symbol, Number)
List = list
Exp = (Atom, List)
# Env = dict


class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."

    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer

    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)


class Procedure(object):
    "A user-defined Scheme procedure"

    def __init__(self, parms, body, env) -> None:
        self.parms, self.body, self.env = parms, body, env

    def __call__(self, *args: Any) -> Any:
        return eval(self.body, Env(self.parms, args, self.env))


def standard_env() -> Env:
    "An environment with some Scheme standard procedures."
    env = Env()
    env.update(vars(math))  # sin, cos, sqrt, pi, ...
    env.update(
        {
            "+": op.add,
            "-": op.sub,
            "*": op.mul,
            "/": op.truediv,
            ">": op.gt,
            "<": op.lt,
            ">=": op.ge,
            "<=": op.le,
            "=": op.eq,
            "abs": abs,
            "append": op.add,
            "apply": lambda proc, args: proc(*args),
            "begin": lambda *x: x[-1],
            "car": lambda x: x[0],
            "cdr": lambda x: x[1:],
            "cons": lambda x, y: [x] + y,
            "eq?": op.is_,
            "expt": pow,
            "equal?": op.eq,
            "length": len,
            "list": lambda *x: List(x),
            "list?": lambda x: isinstance(x, List),
            "map": map,
            "max": max,
            "min": min,
            "not": op.not_,
            "null?": lambda x: x == [],
            "number?": lambda x: isinstance(x, Number),
            "print": print,
            "procedure?": callable,
            "round": round,
            "symbol?": lambda x: isinstance(x, Symbol),
        }
    )

    return env


global_env = standard_env()


def tokenize(chars: str) -> list:
    "Convert a string of characters into a list of tokens."
    return chars.replace("(", " ( ").replace(")", " ) ").split()

=======
import math
import operator as op

def tokenize(chars: str) -> list:
    "Convert a string of characters into a list of tokens."
    return chars.replace('(', ' ( ').replace(')', ' ) ').split()

Symbol = str              # A Scheme Symbol is implemented as a Python str
Number = (int, float)     # A Scheme Number is implemented as a Python int or float
Atom   = (Symbol, Number) # A Scheme Atom is a Symbol or Number
List   = list             # A Scheme List is implemented as a Python list
Exp    = (Atom, List)     # A Scheme expression is an Atom or List
#Env    = dict             # A Scheme environment (defined below) 
#                          # is a mapping of {variable: value}
>>>>>>> 956800b5317dc963faf521b94d6ba1e27eef016a

def parse(program: str) -> Exp:
    "Read a Scheme expression from a string."
    return read_from_tokens(tokenize(program))

<<<<<<< HEAD

def read_from_tokens(tokens: list) -> Exp:
    "Read an expression from a sequence of tokens."
    if len(tokens) == 0:
        raise SyntaxError("unexpected EOF")
    token = tokens.pop(0)
    if token == "(":
        L = []
        while tokens[0] != ")":
            L.append(read_from_tokens(tokens))
        tokens.pop(0)  # pop off ')'
        return L
    elif token == ")":
        raise SyntaxError("unexpected )")
    else:
        return atom(token)


def atom(token: str) -> Atom:
    "Numbers become numbers; every other token is a symbol."
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return Symbol(token)


def eval(x: Exp, env=global_env) -> Exp:
    "Evaluate an expression in an environment."
    if isinstance(x, Symbol):  # variable reference
        return env[x]
    elif isinstance(x, Number):  # constant number
        return x
    op, *args = x

    if op == "quote":  #
        return args[0]
    elif op == "if":  # conditional
        (_, test, conseq, alt) = x
        exp = conseq if eval(test, env) else alt
        return eval(exp, env)
    elif op == "define":  # definition
        (_, symbol, exp) = x
        env[symbol] = eval(exp, env)
    elif op == "lambda":
        (parms, body) = args
        return Procedure(parms, body, env)
    else:  # procedure call
        proc = eval(x[0], env)
        args = [eval(arg, env) for arg in x[1:]]
        return proc(*args)


def scheme_str(exp):
    "Convert a Python object back into a Scheme-readable string."
    if isinstance(exp, List):
        return "(" + " ".join(map(scheme_str, exp)) + ")"
    else:
        return str(exp)


def test_tokenize():
    program = "(begin (define r 10) (* pi (* r r)))"
    assert tokenize(program) == [
        "(",
        "begin",
        "(",
        "define",
        "r",
        "10",
        ")",
        "(",
        "*",
        "pi",
        "(",
        "*",
        "r",
        "r",
        ")",
        ")",
        ")",
    ]


def repl(prompt="lis.py> "):
    while True:
        val = eval(parse(input(prompt)))
        if val is not None:
            print(scheme_str(val))


def test_parse():
    program = "(begin (define r 10) (* pi (* r r)))"
    assert parse(program) == [
        "begin",
        ["define", "r", 10],
        ["*", "pi", ["*", "r", "r"]],
    ]


def test_globl_env():
    assert global_env["+"] == op.add
    assert global_env["sin"] == math.sin


def test_eval():
    assert eval(parse("(begin (define r 10) (* pi (* r r)))")) == 314.1592653589793


def test_scheme_str():
    ex = [
        "begin",
        ["define", "r", 10],
        ["*", "pi", ["*", "r", "r"]],
    ]
    assert scheme_str(ex) == "(begin (define r 10) (* pi (* r r)))"
=======
def read_from_tokens(tokens: list) -> Exp:
    "Read an expression from a sequence of tokens."
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF')
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0) # pop off ')'
        return L
    elif token == ')':
        raise SyntaxError('unexpected )')
    else:
        return atom(token)

def atom(token: str) -> Atom:
    "Numbers become numbers; every other token is a symbol."
    try: return int(token)
    except ValueError:
        try: return float(token)
        except ValueError:
            return Symbol(token)

class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)

def standard_env() -> Env:
    "An environment with some Scheme standard procedures."
    env = Env()
    env.update(vars(math)) # sin, cos, sqrt, pi, ...
    env.update({
        '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv, 
        '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq, 
        'abs':     abs,
        'append':  op.add,  
        'apply':   lambda proc, args: proc(*args),
        'begin':   lambda *x: x[-1],
        'car':     lambda x: x[0],
        'cdr':     lambda x: x[1:], 
        'cons':    lambda x,y: [x] + y,
        'eq?':     op.is_, 
        'expt':    pow,
        'equal?':  op.eq, 
        'length':  len, 
        'list':    lambda *x: List(x), 
        'list?':   lambda x: isinstance(x, List), 
        'map':     map,
        'max':     max,
        'min':     min,
        'not':     op.not_,
        'null?':   lambda x: x == [], 
        'number?': lambda x: isinstance(x, Number),  
		'print':   print,
        'procedure?': callable,
        'round':   round,
        'symbol?': lambda x: isinstance(x, Symbol),
    })
    return env

global_env = standard_env()

def eval(x, env=global_env):
    "Evaluate an expression in an environment."
    if isinstance(x, Symbol):    # variable reference
        return env.find(x)[x]
    elif not isinstance(x, List):# constant 
        return x   
    op, *args = x       
    if op == 'quote':            # quotation
        return args[0]
    elif op == 'if':             # conditional
        (test, conseq, alt) = args
        exp = (conseq if eval(test, env) else alt)
        return eval(exp, env)
    elif op == 'define':         # definition
        (symbol, exp) = args
        env[symbol] = eval(exp, env)
    elif op == 'set!':           # assignment
        (symbol, exp) = args
        env.find(symbol)[symbol] = eval(exp, env)
    elif op == 'lambda':         # procedure
        (parms, body) = args
        return Procedure(parms, body, env)
    else:                        # procedure call
        proc = eval(op, env)
        vals = [eval(arg, env) for arg in args]
        return proc(*vals)

def schemestr(exp):
    "Convert a Python object back into a Scheme-readable string."
    if isinstance(exp, List):
        return '(' + ' '.join(map(schemestr, exp)) + ')' 
    else:
        return str(exp)

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    def __call__(self, *args): 
        return eval(self.body, Env(self.parms, args, self.env))

def repl(prompt='lis.py> '):
    "A prompt-read-eval-print loop."
    while True:
        val = eval(parse(input(prompt)))
        if val is not None: 
            print(schemestr(val))

def test_tokenize():
    program = "(begin (define r 10) (* pi (* r r)))"
    assert tokenize(program) == ['(', 'begin', '(', 'define', 'r', '10', ')', '(', '*', 'pi', '(', '*', 'r', 'r', ')', ')', ')']

def test_parse():
    program = "(begin (define r 10) (* pi (* r r)))"
    assert parse(program) == ['begin', ['define', 'r', 10], ['*', 'pi', ['*', 'r', 'r']]]

def test_global_env():
    assert global_env["+"] == op.add
    assert global_env["sin"] == math.sin

def test_eval():
    program = "(begin (define r 10) (* pi (* r r)))"
    assert(eval(parse(program))) == 314.1592653589793

def test_schemestr():
    expr = ['begin', ['define', 'r', 10], ['*', 'pi', ['*', 'r', 'r']]]
    assert schemestr(expr) == "(begin (define r 10) (* pi (* r r)))"
>>>>>>> 956800b5317dc963faf521b94d6ba1e27eef016a


if __name__ == "__main__":
    test_tokenize()
    test_parse()
<<<<<<< HEAD
    test_globl_env()
    test_scheme_str()
    repl()
=======
    test_global_env()
    test_eval()
    test_schemestr()
    print("done")
    repl()
>>>>>>> 956800b5317dc963faf521b94d6ba1e27eef016a
