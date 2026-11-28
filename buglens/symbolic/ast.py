from dataclasses import dataclass
from typing import Union
import math


class Expr:
    __slots__ = ()


@dataclass(frozen=True, slots=True)
class Num(Expr):
    value: Union[int, float]

    def __post_init__(self):
        if isinstance(self.value, bool):
            raise TypeError("Boolean values are not allowed")

        if not isinstance(self.value, (int, float)):
            raise TypeError("Number must be an integer or float")

        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("Number must be finite")


@dataclass(frozen=True, slots=True)
class Var(Expr):
    name: str


@dataclass(frozen=True, slots=True)
class Neg(Expr):
    child: Expr


@dataclass(frozen=True, slots=True)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr

    def __post_init__(self):
        if self.op not in {"+", "-", "*", "/", "**"}:
            raise ValueError("Unsupported operator")


@dataclass(frozen=True, slots=True)
class Frac(Expr):
    num: Expr
    den: Expr


@dataclass(frozen=True, slots=True)
class EqNode(Expr):
    lhs: Expr
    rhs: Expr