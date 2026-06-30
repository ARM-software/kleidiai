#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Render micro-kernel naming documentation."""
from __future__ import annotations

from collections.abc import Iterable

import naming.rules as naming
from naming.grammar import Doc
from naming.grammar import Expr
from naming.grammar import OneOf
from naming.grammar import Rule
from naming.grammar import Seq


def _grammar_rule_blocks() -> str:
    return _rule_definition_blocks(
        reversed(tuple(naming.grammar.iter_documented_rules()))
    )


def _rule_heading(rule: Rule) -> str:
    if rule.title:
        return f"### {rule.title}"
    return f"### `{rule.name}`"


def _sentence(text: str) -> str:
    if not text or text[-1] in ".!?":
        return text
    return f"{text}."


def _rule_definition_blocks(rules: Iterable[Rule]) -> str:
    lines = []
    for rule in rules:
        if rule.title:
            lines.extend((_rule_heading(rule), ""))
        if rule.description:
            lines.extend((_sentence(rule.description), ""))
        lines.extend(
            (
                f"**`{rule.name}`** = `{rule.expr.to_grammar()}`",
                "",
            )
        )
        documented_items = tuple(_documented_items(rule.expr))
        if documented_items:
            lines.extend(("where:", ""))
            lines.extend(
                f"- **`{grammar}`**: {description}"
                for grammar, description in documented_items
            )
            lines.append("")
    return "\n".join(lines).rstrip()


def _documented_items(expr: Expr) -> Iterable[tuple[str, str]]:
    if isinstance(expr, OneOf):
        items = expr.options
    elif isinstance(expr, Seq):
        items = expr.items
    else:
        return

    for item in items:
        if isinstance(item, Doc):
            yield (item.to_grammar(), item.description)


_COMMON_NAME_SHAPES = """### Common Name Shapes

- LHS packing micro-kernels are named `kai_lhs_pack_<output>_<input>_<description>`.
- RHS packing micro-kernels are named `kai_rhs_pack_<orientation>_<output>_<inputs>_<description>`.
- Matmul compute micro-kernels are named `kai_<operation>_<output>_<LHS input>_<RHS input>_<description>`.
- Depthwise convolution micro-kernels are named with the depthwise operation, buffers, filter, stride, output block, SIMD engine, and optional instruction.

For matmul-family compute micro-kernels, buffer descriptors appear in the order
destination, LHS input, then RHS input. The output descriptors of LHS and RHS
packing micro-kernels match the corresponding packed input descriptors of the
matmul micro-kernel.

### Packed Buffer Layouts

Packed buffers include a `p<width>x<height>` layout in full micro-kernel names.
For matmul-family names, the LHS packed width is normally the row blocking
dimension (`MR`), and the RHS packed width is normally the column blocking
dimension (`NR`). The packed height is the block depth (`BD`), which is derived
from the `KR` and `SR` values used by the micro-kernel as `KR / SR`.

Packed buffers can also encode data order, packed scale type, and packed bias
type. Scale values are encoded as `s<type>` and bias values as `b<type>`.
Legacy `scalef*` and `biasf*` spellings are not valid in the grammar."""


def render_documentation() -> str:
    """Render the generated micro-kernel naming documentation."""
    grammar = naming.grammar.to_grammar()
    grammar_rules = _grammar_rule_blocks()

    return f"""<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# The micro-kernel naming scheme

This document describes the KleidiAI micro-kernel naming scheme.

## Naming Structure

The high level view of micro-kernel naming is generated from the documented
grammar rules:

- Micro-kernel source files use names beginning with `kai_`.
- Matmul micro-kernels use the `matmul_ukernel_name` grammar.
- Depthwise convolution micro-kernels describe the operation, buffers, filter, stride, output block, SIMD engine, and optional instruction through the `dwconv_ukernel_name` grammar.
- Micro-kernel directories use the `directory_name` grammar.

{_COMMON_NAME_SHAPES}

### How to Read the Grammar

- **`rule_name`**: Reference to another grammar rule.
- **`"text"`**: Literal text that appears in the name.
- **`a b`**: Sequence: `a` followed by `b`.
- **`a | b`**: Choice: either `a` or `b`.
- **`[a]`**: Optional expression.
- **`a+`**: One or more repetitions of `a`.
- **`(a b)`**: Grouped expression.
- **`@natural_int`**: Positive integer literal.
- **`@pow2_int`**: Power-of-two integer literal.
- **`@operand_type`**: Data type descriptor.

## Naming Rules

The documented naming rules are listed below.

{grammar_rules}

## Rule Enforcement

The naming scheme is implemented in `tools/naming/rules.py`. Micro-kernel and
directory names are checked by `tools/check_microkernel_names.py`, and CI uses
the same checker to enforce the rules.

This document is generated from the naming rules. Regenerate it with
`tools/check_microkernel_names.py --generate-documentation` after updating the
naming scheme.

## Known naming issues

Some legacy micro-kernel and directory names do not match the current grammar,
but are retained to preserve the existing API. The known exceptions are listed
in `tools/naming/issues.py`. To include them in checker output, run
`tools/check_microkernel_names.py --report-known-issues`.

## Full naming grammar

The grammar below is generated from the naming rules.

```text
{grammar}
```
"""
