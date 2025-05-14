"""
Structural FX for scripts.

References:
    https://onlinelibrary.wiley.com/doi/epdf/10.1155/2021/9923234
    https://dl.acm.org/doi/pdf/10.1145/3627106.3627138
"""

import ast
import inspect
import sys
import zlib
from typing import Any, Callable

import numpy as np
import validators
from radon.complexity import cc_visit

FXType = Callable[[str], Any]

# fmt: off
AST_NODES = [
    'AST', 'Add', 'And', 'AnnAssign', 'Assert', 'Assign', 'AsyncFor',
    'AsyncFunctionDef', 'AsyncWith', 'Attribute', 'AugAssign', 'AugLoad',
    'AugStore', 'Await', 'BinOp', 'BitAnd', 'BitOr', 'BitXor', 'BoolOp',
    'Break', 'Bytes', 'Call', 'ClassDef', 'Compare', 'Constant', 'Continue',
    'Del', 'Delete', 'Dict', 'DictComp', 'Div', 'Ellipsis', 'Eq', 'ExceptHandler',
    'Expr', 'Expression', 'ExtSlice', 'FloorDiv', 'For', 'FormattedValue', 'FunctionDef',
    'FunctionType', 'GeneratorExp', 'Global', 'Gt', 'GtE', 'If', 'IfExp', 'Import', 'ImportFrom',
    'In', 'Index', 'Interactive', 'Invert', 'Is', 'IsNot', 'JoinedStr', 'LShift', 'Lambda', 'List',
    'ListComp', 'Load', 'Lt', 'LtE', 'MatMult', 'Match', 'MatchAs', 'MatchClass', 'MatchMapping',
    'MatchOr', 'MatchSequence', 'MatchSingleton', 'MatchStar', 'MatchValue', 'Mod', 'Module', 'Mult',
    'Name', 'NameConstant', 'NamedExpr', 'Nonlocal', 'Not', 'NotEq', 'NotIn', 'Num', 'Or', 'Param', 'Pass',
    'Pow', 'RShift', 'Raise', 'Return', 'Set', 'SetComp', 'Slice', 'Starred', 'Store', 'Str', 'Sub', 'Subscript',
    'Suite', 'Try', 'TryStar', 'Tuple', 'TypeIgnore', 'UAdd', 'USub', 'UnaryOp', 'While', 'With', 'Yield', 'YieldFrom',
    'alias', 'arg', 'arguments', 'boolop', 'cmpop', 'comprehension', 'excepthandler', 'expr', 'expr_context', 'keyword',
    'match_case', 'mod', 'operator', 'pattern', 'slice', 'stmt', 'type_ignore', 'unaryop', 'withitem'
]
assert len(AST_NODES) == 131 # with python3.11
# fmt: on


def collect_all_fxs() -> list[str]:
    """
    Exposes all fx_* functions from this module so that we can combine them.
    """
    return [
        f for f, obj in inspect.getmembers(sys.modules[__name__]) if inspect.isfunction(obj) and f.startswith("fx_")
    ]


def can_be_parsed(filepath: str) -> bool:
    try:
        with open(filepath, "rt") as fp:
            ast.parse(fp.read())
        return True
    except SyntaxError:
        return False


def fx_ast_node_count(source: str) -> list:
    """
    Count all known AST node types.
    """
    tree = ast.parse(source)

    cnt = {node: 0 for node in AST_NODES}
    for node in ast.walk(tree):
        if (n := node.__class__.__name__) in cnt:
            cnt[n] += 1

    return [cnt[n] for n in AST_NODES]


def fx_count_lines(source: str) -> int:
    return source.count("\n")


def fx_cyclomatic_complexity(source: str) -> float:
    """
    Ref. https://en.wikipedia.org/wiki/Cyclomatic_complexity
    """
    res = cc_visit(source)
    if not res:
        return 0
    return sum(x.complexity for x in res) / len(res)


def fx_string_stats(source: str) -> list:
    """
    Collect string and bytes and compute: [count, min, max, mean, 4-quantiles]
    """

    class StrCollector(ast.NodeVisitor):
        def __init__(self):
            self.acc = []

        def visit_Str(self, x):
            self.acc.append(len(x.value))

        def visit_Bytes(self, x):
            self.acc.append(len(x.value))

    cnt = StrCollector()
    cnt.visit(ast.parse(source))

    if not cnt.acc:
        return [np.nan] * 8

    return [
        len(cnt.acc),
        min(cnt.acc),
        max(cnt.acc),
        np.mean(cnt.acc),
        *np.quantile(cnt.acc, [0.25, 0.5, 0.75, 0.9]),
    ]


def fx_compression_ratio(source: str) -> float:
    """
    Not entropy per se, but rather a proxy via how compressable the source is.
    Lower values mean less entropy in the file and less likely to contain random strings.
    """
    xs = bytes(source, encoding="utf8")
    if len(xs) == 0:
        return np.nan  # "nothing" to compress
    return len(zlib.compress(xs)) / len(xs)


def fx_base64_count(source: str) -> int:
    """
    Counts how many strings in the source are valid base64 encodings.
    """

    class Base64Count(ast.NodeVisitor):
        def __init__(self):
            self.acc = []

        def visit_Str(self, x):
            x = x.value
            if validators.base64(x):
                self.acc.append(x)

        def visit_Bytes(self, x):
            x = x.value
            try:
                if validators.base64(x.decode()):
                    self.acc.append(x)
            except UnicodeDecodeError:
                return

    cnt = Base64Count()
    cnt.visit(ast.parse(source))
    return len(cnt.acc)


def fx_url_count(source: str) -> int:
    """
    Count urls in the source
    """

    class URLCount(ast.NodeVisitor):
        def __init__(self):
            self.acc = []

        def visit_Str(self, x):
            x = x.value

            if validators.url(x):
                self.acc.append(x)

    cnt = URLCount()
    cnt.visit(ast.parse(source))
    return len(cnt.acc)


def fx_ip_count(source: str) -> int:
    """
    Count IPv4 and IPv6 in the source
    """

    class IPCount(ast.NodeVisitor):
        def __init__(self):
            self.acc = []

        def visit_Str(self, x):
            x = x.value
            if validators.ipv4(x) or validators.ipv6(x):
                self.acc.append(x)

    cnt = IPCount()
    cnt.visit(ast.parse(source))
    return len(cnt.acc)
