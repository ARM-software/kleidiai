#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers for describing grammars as Python objects.

This module provides a small expression system for writing grammar rules in a
readable Python form. It is intended only for documentation generation and name
validation. Rules are built from ``Expr`` objects:

* ``Seq`` matches expressions in order.
* ``OneOf`` matches one expression from a set of alternatives.
* ``Optional`` matches an expression if it is present.
* ``OneOrMore`` matches repeated expressions.
* ``Literal`` matches a literal string.
* ``NaturalInt`` matches a positive integer.
* ``Pow2Int`` matches a natural integer that is a power of 2 (1, 2, 4, 8...)
* ``OperandType`` matches a data type descriptor (f32, i8, bf16...)

``Grammar`` is a local rule registry. Its ``rule`` decorator evaluates the
decorated function once, wraps the returned expression in a named ``Rule``, and
stores that rule for matching and documentation generation. Only rules with a
description are included by ``render_markdown``.

Matching is position based. Every expression matches against an input string
from a starting offset and returns either a ``MatchSuccess`` or ``MatchFailure``.
``MatchSuccess.pos`` contains the end offset after a successful match.
Alternatives use the longest successful match. Equal-length successful
alternatives are invalid grammar.

* ``MatchFailure.failures`` contains ``MatchExpectation`` entries with the input
    offset and expected grammar fragment used for developer-facing diagnostics.
* ``Expr.match_full`` converts partial matches into a failure that expects the
    end of input.
* ``furthest_failure`` selects the most advanced failure position and combines
    the expected fragments reported there.
"""
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class MatchExpectation:
    """An expected grammar fragment at a failed input position."""

    pos: int
    expected: str


@dataclass(frozen=True)
class MatchSuccess:
    pos: int

    @property
    def matched(self) -> bool:
        return True

    def furthest_failure(self) -> None:
        return None


@dataclass
class MatchFailure:
    failures: list[MatchExpectation]

    @classmethod
    def expected(cls, pos: int, expected: str) -> "MatchFailure":
        return cls([MatchExpectation(pos, expected)])

    @property
    def matched(self) -> bool:
        return False

    def furthest_failure(self) -> MatchExpectation | None:
        if not self.failures:
            return None

        max_pos = max(failure.pos for failure in self.failures)
        expected = " | ".join(
            dict.fromkeys(
                failure.expected for failure in self.failures if failure.pos == max_pos
            )
        )
        return MatchExpectation(max_pos, expected)


MatchResult: TypeAlias = MatchSuccess | MatchFailure


# -----------------------------------------------------------------------------
# Grammar building blocks
# -----------------------------------------------------------------------------


class Expr(ABC):
    """Base class for grammar expressions."""

    @property
    def compound(self) -> bool:
        """Whether this expression should be grouped when embedded."""
        return False

    def match_full(self, text: str) -> MatchResult:
        """Used for top level expressions to ensure that grammar fully matches a string"""
        result = self.match(text)
        if result.matched and result.pos != len(text):
            return MatchFailure.expected(result.pos, "end of input")
        return result

    @abstractmethod
    def match(self, text: str, pos: int = 0) -> MatchResult:
        """Match this expression against ``text`` starting at ``pos``."""
        raise NotImplementedError

    @abstractmethod
    def to_grammar(self) -> str:
        """Render this expression as grammar documentation."""
        raise NotImplementedError

    def __str__(self) -> str:
        return self.to_grammar()


class _CompoundExpr(Expr):
    """Base class for grammar expressions that need grouping when embedded."""

    @property
    def compound(self) -> bool:
        """Whether this expression should be grouped when embedded."""
        return True


class Seq(_CompoundExpr):
    """A sequence of grammar expressions."""

    def __init__(self, *items: Expr | str) -> None:
        self.items = tuple(_as_expr(item) for item in items)

    def match(self, text: str, pos: int = 0) -> MatchResult:
        position = pos
        for item in self.items:
            result = item.match(text, position)
            if not result.matched:
                return result
            position = result.pos

        return MatchSuccess(position)

    def to_grammar(self) -> str:
        return " ".join(
            f"({item.to_grammar()})" if item.compound else item.to_grammar()
            for item in self.items
        )


class OneOf(_CompoundExpr):
    """A choice between multiple grammar expressions."""

    def __init__(self, *options: Expr | str) -> None:
        self.options = tuple(_as_expr(option) for option in options)

    def match(self, text: str, pos: int = 0) -> MatchResult:
        matches = []
        failures = []
        for option in self.options:
            result = option.match(text, pos)
            if result.matched:
                matches.append(result)
            else:
                failures.extend(result.failures)

        if matches:
            best_pos = max(result.pos for result in matches)
            best_matches = [result for result in matches if result.pos == best_pos]
            assert len(best_matches) == 1, (
                "Ambiguous grammar alternatives consume the same input: "
                f"{self.to_grammar()}"
            )
            return MatchSuccess(best_pos)

        result = MatchFailure(failures)
        failure = result.furthest_failure()
        assert failure is not None
        return MatchFailure.expected(failure.pos, failure.expected)

    def to_grammar(self) -> str:
        return " | ".join(option.to_grammar() for option in self.options)


class Optional(Expr):
    """An optional grammar expression."""

    def __init__(self, expr: Expr | str) -> None:
        self.expr = _as_expr(expr)

    def match(self, text: str, pos: int = 0) -> MatchResult:
        result = self.expr.match(text, pos)
        if result.matched:
            assert result.pos != pos, "Optional expression matched empty input"
            return MatchSuccess(result.pos)
        return MatchSuccess(pos)

    def to_grammar(self) -> str:
        return f"[{self.expr.to_grammar()}]"


class OneOrMore(Expr):
    """A grammar expression that must appear at least once."""

    def __init__(self, expr: Expr | str) -> None:
        self.expr = _as_expr(expr)

    def match(self, text: str, pos: int = 0) -> MatchResult:
        result = self.expr.match(text, pos)
        if not result.matched:
            return result

        assert result.pos != pos, "Repeated expression matched empty input"
        position = result.pos
        while True:
            result = self.expr.match(text, position)
            if not result.matched:
                return MatchSuccess(position)
            assert result.pos != position, "Repeated expression matched empty input"
            position = result.pos

    def to_grammar(self) -> str:
        if not self.expr.compound:
            return f"{self.expr.to_grammar()}+"
        return f"({self.expr.to_grammar()})+"


class Literal(Expr):
    """A literal string in the grammar."""

    def __init__(self, value: str) -> None:
        self.value = value

    def match(self, text: str, pos: int = 0) -> MatchResult:
        if text.startswith(self.value, pos):
            return MatchSuccess(pos + len(self.value))
        return MatchFailure.expected(pos, self.to_grammar())

    def to_grammar(self) -> str:
        return f'"{self.value}"'


class NaturalInt(Expr):
    """A positive integer literal."""

    def match(self, text: str, pos: int = 0) -> MatchResult:
        return _match_positive_integer(text, pos)

    def to_grammar(self) -> str:
        return "@natural_int"


class Pow2Int(Expr):
    """A power-of-two integer literal."""

    def match(self, text: str, pos: int = 0) -> MatchResult:
        return _match_positive_integer(
            text, pos, lambda value: value > 0 and value & (value - 1) == 0
        )

    def to_grammar(self) -> str:
        return "@pow2_int"


class OperandType(Expr):
    """A data type descriptor."""

    def match(self, text: str, pos: int = 0) -> MatchResult:
        for base_type in ("bf", "f", "i", "u", "x"):
            if text.startswith(base_type, pos):
                result = _match_positive_integer(
                    text,
                    pos + len(base_type),
                    lambda value: value > 0 and value & (value - 1) == 0,
                )
                if result.matched:
                    return result
                return MatchFailure.expected(pos, self.to_grammar())

        return MatchFailure.expected(pos, self.to_grammar())

    def to_grammar(self) -> str:
        return "@operand_type"


# -----------------------------------------------------------------------------
# Rule registration and documentation rendering
# -----------------------------------------------------------------------------


class Rule(Expr):
    """A named grammar rule with a description."""

    def __init__(self, name: str, expr: Expr, description: str | None) -> None:
        self.name = name
        self.expr = expr
        self.description = description

    def match(self, text: str, pos: int = 0) -> MatchResult:
        return self.expr.match(text, pos)

    def to_grammar(self) -> str:
        return self.name

    def definition(self) -> str:
        return f"{self.name} = {self.expr.to_grammar()}"

    def to_markdown(self) -> str:
        lines = [
            f"### `{self.name}`",
            "",
            "```text",
            self.definition(),
            "```",
        ]
        if self.description:
            lines.extend(("", self.description))
        return "\n".join(lines)


class Grammar:
    """A local registry for grammar rules."""

    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}

    def rule(
        self, description: str | None = None, *, name: str | None = None
    ) -> Callable[[Callable[[], Expr | str]], Rule]:
        def decorate(fn: Callable[[], Expr | str]) -> Rule:
            rule_name = name or fn.__name__
            if rule_name in self._rules:
                raise ValueError(f"Grammar rule already registered: {rule_name}")

            rule = Rule(rule_name, _as_expr(fn()), description)
            self._rules[rule_name] = rule
            return rule

        return decorate

    def __iter__(self) -> Iterator[Rule]:
        return iter(self.iter_rules())

    def iter_rules(self) -> Iterable[Rule]:
        return self._rules.values()

    def iter_documented_rules(self) -> Iterable[Rule]:
        return (rule for rule in self.iter_rules() if rule.description)

    def to_grammar(self) -> str:
        return "\n".join(rule.definition() for rule in reversed(self._rules.values()))

    def render_markdown(self) -> str:
        return "\n\n".join(rule.to_markdown() for rule in self.iter_documented_rules())


def _as_expr(value: Expr | str) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, str):
        return Literal(value)
    raise TypeError(f"Expected grammar expression or string literal, got {type(value)}")


def _match_positive_integer(
    text: str, pos: int, predicate: Callable[[int], bool] | None = None
) -> MatchResult:
    expected = "@natural_int" if predicate is None else "@pow2_int"
    if pos >= len(text) or not text[pos].isdigit() or text[pos] == "0":
        return MatchFailure.expected(pos, expected)

    end = pos
    while end < len(text) and text[end].isdigit():
        end += 1

    value = int(text[pos:end])
    if predicate is None or predicate(value):
        return MatchSuccess(end)
    return MatchFailure.expected(pos, expected)
