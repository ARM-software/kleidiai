#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers for describing grammars as Python objects.

This module provides a small expression system for writing grammar rules in a
readable Python form. It is intended only for documentation generation and name
validation. Rules are built from ``Expr`` objects:

* ``Seq`` parses expressions in order.
* ``OneOf`` parses one expression from a set of alternatives.
* ``Optional`` parses an expression if it is present.
* ``OneOrMore`` parses repeated expressions.
* ``Literal`` parses a literal string.
* ``Doc`` attaches documentation to an expression without changing it.
* ``NaturalInt`` parses a positive integer.
* ``Pow2Int`` parses a natural integer that is a power of 2 (1, 2, 4, 8...)
* ``OperandType`` parses a data type descriptor (f32, i8, bf16...)

``Grammar`` is a local rule registry. Its ``rule`` decorator evaluates the
decorated function once, wraps the returned expression in a named ``Rule``, and
stores that rule for parsing and documentation generation. Only rules with a
title are included by ``render_markdown``.

Parsing is position based. Every expression parses an input string
from a starting offset and returns either a ``ParseSuccess`` or ``ParseFailure``.
``ParseSuccess.pos`` contains the end offset after a successful parse.
``ParseSuccess.descriptions`` contains descriptions from parsed ``Doc`` nodes.
Alternatives use the longest successful parse. Equal-length successful
alternatives are invalid grammar.

* ``ParseFailure.failures`` contains ``ParseExpectation`` entries with the input
    offset and expected grammar fragment used for developer-facing diagnostics.
* ``Expr.parse_full`` converts partial parses into a failure that expects the
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
from dataclasses import field
from typing import TypeAlias


@dataclass(frozen=True)
class ParseExpectation:
    """An expected grammar fragment at a failed input position."""

    pos: int
    expected: str


@dataclass(frozen=True)
class ParseSuccess:
    pos: int
    descriptions: list[tuple[int, str]] = field(default_factory=list)

    @property
    def parsed_ok(self) -> bool:
        return True

    def furthest_failure(self) -> None:
        return None


@dataclass
class ParseFailure:
    failures: list[ParseExpectation]

    @classmethod
    def expected(cls, pos: int, expected: str) -> "ParseFailure":
        return cls([ParseExpectation(pos, expected)])

    @property
    def parsed_ok(self) -> bool:
        return False

    def furthest_failure(self) -> ParseExpectation | None:
        if not self.failures:
            return None

        max_pos = max(failure.pos for failure in self.failures)
        expected = " | ".join(
            dict.fromkeys(
                failure.expected for failure in self.failures if failure.pos == max_pos
            )
        )
        return ParseExpectation(max_pos, expected)


ParseResult: TypeAlias = ParseSuccess | ParseFailure


# -----------------------------------------------------------------------------
# Grammar building blocks
# -----------------------------------------------------------------------------


class Expr(ABC):
    """Base class for grammar expressions."""

    @property
    def compound(self) -> bool:
        """Whether this expression should be grouped when embedded."""
        return False

    def parse_full(self, text: str) -> ParseResult:
        """Used for top level expressions to ensure that grammar fully parses a string"""
        result = self.parse(text)
        if result.parsed_ok and result.pos != len(text):
            return ParseFailure.expected(result.pos, "end of input")
        return result

    @abstractmethod
    def parse(self, text: str, pos: int = 0) -> ParseResult:
        """Parse this expression against ``text`` starting at ``pos``."""
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

    def parse(self, text: str, pos: int = 0) -> ParseResult:
        position = pos
        descriptions: list[tuple[int, str]] = []
        for item in self.items:
            result = item.parse(text, position)
            if not result.parsed_ok:
                return result
            descriptions.extend(result.descriptions)
            position = result.pos

        return ParseSuccess(position, descriptions)

    def to_grammar(self) -> str:
        return " ".join(
            f"({item.to_grammar()})" if item.compound else item.to_grammar()
            for item in self.items
        )


class OneOf(_CompoundExpr):
    """A choice between multiple grammar expressions."""

    def __init__(self, *options: Expr | str) -> None:
        self.options = tuple(_as_expr(option) for option in options)

    def parse(self, text: str, pos: int = 0) -> ParseResult:
        parses: list[ParseSuccess] = []
        failures = []
        for option in self.options:
            result = option.parse(text, pos)
            if result.parsed_ok:
                parses.append(result)
            else:
                failures.extend(result.failures)

        if parses:
            best_pos = max(result.pos for result in parses)
            best_parses = [result for result in parses if result.pos == best_pos]
            assert len(best_parses) == 1, (
                "Ambiguous grammar alternatives consume the same input: "
                f"{self.to_grammar()}"
            )
            return best_parses[0]

        result = ParseFailure(failures)
        failure = result.furthest_failure()
        assert failure is not None
        return ParseFailure.expected(failure.pos, failure.expected)

    def to_grammar(self) -> str:
        return " | ".join(option.to_grammar() for option in self.options)


class Optional(Expr):
    """An optional grammar expression."""

    def __init__(self, expr: Expr | str) -> None:
        self.expr = _as_expr(expr)

    def parse(self, text: str, pos: int = 0) -> ParseResult:
        result = self.expr.parse(text, pos)
        if result.parsed_ok:
            assert result.pos != pos, "Optional expression parsed empty input"
            return result
        return ParseSuccess(pos)

    def to_grammar(self) -> str:
        return f"[{self.expr.to_grammar()}]"


class OneOrMore(Expr):
    """A grammar expression that must appear at least once."""

    def __init__(self, expr: Expr | str) -> None:
        self.expr = _as_expr(expr)

    def parse(self, text: str, pos: int = 0) -> ParseResult:
        result = self.expr.parse(text, pos)
        if not result.parsed_ok:
            return result

        assert result.pos != pos, "Repeated expression parsed empty input"
        position = result.pos
        descriptions = list(result.descriptions)
        while True:
            result = self.expr.parse(text, position)
            if not result.parsed_ok:
                return ParseSuccess(position, descriptions)
            assert result.pos != position, "Repeated expression parsed empty input"
            descriptions.extend(result.descriptions)
            position = result.pos

    def to_grammar(self) -> str:
        if not self.expr.compound:
            return f"{self.expr.to_grammar()}+"
        return f"({self.expr.to_grammar()})+"


class Literal(Expr):
    """A literal string in the grammar."""

    def __init__(self, value: str) -> None:
        self.value = value

    def parse(self, text: str, pos: int = 0) -> ParseResult:
        if text.startswith(self.value, pos):
            return ParseSuccess(pos + len(self.value))
        return ParseFailure.expected(pos, self.to_grammar())

    def to_grammar(self) -> str:
        return f'"{self.value}"'


class Doc(Expr):
    """Documentation for a grammar expression."""

    def __init__(self, expr: Expr | str, *, description: str) -> None:
        self.expr = _as_expr(expr)
        self.description = description

    @property
    def compound(self) -> bool:
        return self.expr.compound

    def parse(self, text: str, pos: int = 0) -> ParseResult:
        result = self.expr.parse(text, pos)
        if result.parsed_ok and result.pos != pos:
            return ParseSuccess(
                result.pos,
                [(pos, self.description), *result.descriptions],
            )
        return result

    def to_grammar(self) -> str:
        return self.expr.to_grammar()


class NaturalInt(Expr):
    """A positive integer literal."""

    def parse(self, text: str, pos: int = 0) -> ParseResult:
        return _parse_positive_integer(text, pos)

    def to_grammar(self) -> str:
        return "@natural_int"


class Pow2Int(Expr):
    """A power-of-two integer literal."""

    def parse(self, text: str, pos: int = 0) -> ParseResult:
        return _parse_positive_integer(
            text, pos, lambda value: value > 0 and value & (value - 1) == 0
        )

    def to_grammar(self) -> str:
        return "@pow2_int"


class OperandType(Expr):
    """A data type descriptor."""

    def parse(self, text: str, pos: int = 0) -> ParseResult:
        for base_type in ("bf", "f", "i", "u", "x"):
            if text.startswith(base_type, pos):
                result = _parse_positive_integer(
                    text,
                    pos + len(base_type),
                    lambda value: value > 0 and value & (value - 1) == 0,
                )
                if result.parsed_ok:
                    return result
                return ParseFailure.expected(pos, self.to_grammar())

        return ParseFailure.expected(pos, self.to_grammar())

    def to_grammar(self) -> str:
        return "@operand_type"


# -----------------------------------------------------------------------------
# Rule registration and documentation rendering
# -----------------------------------------------------------------------------


class Rule(Expr):
    """A named grammar rule with documentation."""

    def __init__(
        self,
        name: str,
        expr: Expr,
        title: str | None,
        description: str | None,
    ) -> None:
        self.name = name
        self.expr = expr
        self.title = title
        self.description = description

    def parse(self, text: str, pos: int = 0) -> ParseResult:
        return self.expr.parse(text, pos)

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
        if self.title:
            lines.extend(("", self.title))
        if self.description:
            lines.extend(("", self.description))
        return "\n".join(lines)


class Grammar:
    """A local registry for grammar rules."""

    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}

    def rule(
        self,
        *,
        title: str | None = None,
        description: str | None = None,
        name: str | None = None,
    ) -> Callable[[Callable[[], Expr | str]], Rule]:
        def decorate(fn: Callable[[], Expr | str]) -> Rule:
            rule_name = name or fn.__name__
            if rule_name in self._rules:
                raise ValueError(f"Grammar rule already registered: {rule_name}")

            rule = Rule(rule_name, _as_expr(fn()), title, description)
            self._rules[rule_name] = rule
            return rule

        return decorate

    def __iter__(self) -> Iterator[Rule]:
        return iter(self.iter_rules())

    def iter_rules(self) -> Iterable[Rule]:
        return self._rules.values()

    def iter_documented_rules(self) -> Iterable[Rule]:
        return (rule for rule in self.iter_rules() if rule.title)

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


def _parse_positive_integer(
    text: str, pos: int, predicate: Callable[[int], bool] | None = None
) -> ParseResult:
    expected = "@natural_int" if predicate is None else "@pow2_int"
    if pos >= len(text) or not text[pos].isdigit() or text[pos] == "0":
        return ParseFailure.expected(pos, expected)

    end = pos
    while end < len(text) and text[end].isdigit():
        end += 1

    value = int(text[pos:end])
    if predicate is None or predicate(value):
        return ParseSuccess(end)
    return ParseFailure.expected(pos, expected)
