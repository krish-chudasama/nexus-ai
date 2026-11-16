"""Safe arithmetic evaluator. Parses with ``ast`` and walks an allow-list of nodes;
never calls eval/exec, so arbitrary code execution is impossible by construction."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

from app.core.errors import ValidationFailed

MAX_EXPR_LEN = 200
MAX_NODES = 100
MAX_EXPONENT = 100
MAX_BASE = 1e6
MAX_INT_BITS = 4096

_BIN: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_CONSTS = {"pi": math.pi, "e": math.e}


class _Evaluator:
    def __init__(self) -> None:
        self.nodes = 0

    def eval(self, node: ast.AST) -> float | int:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise ValidationFailed("Expression is too complex.")
        if isinstance(node, ast.Expression):
            return self.eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int | float):
                raise ValidationFailed("Only numbers are allowed.")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = self.eval(node.left), self.eval(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > MAX_EXPONENT or abs(left) > MAX_BASE):
                raise ValidationFailed("Exponent or base is too large.")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](self.eval(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*[self.eval(a) for a in node.args])
        raise ValidationFailed(f"Unsupported syntax: {type(node).__name__}.")


def calculate(expression: str) -> float | int:
    expression = (expression or "").strip()
    if not expression:
        raise ValidationFailed("Expression is empty.")
    if len(expression) > MAX_EXPR_LEN:
        raise ValidationFailed(f"Expression exceeds {MAX_EXPR_LEN} characters.")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _Evaluator().eval(tree)
    except ValidationFailed:
        raise
    except ZeroDivisionError as exc:
        raise ValidationFailed("Division by zero.") from exc
    except (SyntaxError, ValueError, OverflowError, TypeError, RecursionError, MemoryError) as exc:
        raise ValidationFailed(f"Could not evaluate expression: {type(exc).__name__}.") from exc
    if isinstance(result, int) and result.bit_length() > MAX_INT_BITS:
        raise ValidationFailed("Result is too large.")
    if isinstance(result, float):
        if math.isnan(result) or math.isinf(result):
            raise ValidationFailed("Result is not a finite number.")
        result = float(f"{result:.12g}")
    return result
