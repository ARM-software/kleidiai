#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0


def annotate(text: str, descriptions: list[tuple[int, str]]) -> str:
    """Annotate offsets in ``text`` with aligned descriptions."""
    if not descriptions:
        return text

    for offset, _ in descriptions:
        if offset < 0 or offset >= len(text):
            raise ValueError(f"annotation offset out of range: {offset}")

    sorted_descriptions = sorted(descriptions, key=lambda description: description[0])
    offsets = sorted({offset for offset, _ in sorted_descriptions})
    annotated_offsets = set(offsets)

    display_positions: dict[int, int] = {}
    display_text_parts: list[str] = []
    for offset, char in enumerate(text):
        if offset in annotated_offsets and offset != 0:
            display_text_parts.append(" ")

        if offset in annotated_offsets:
            display_positions[offset] = len(display_text_parts)

        display_text_parts.append(char)

    display_text = "".join(display_text_parts)
    display_width = len(display_text)

    lines = [display_text]

    marker_line = [" "] * display_width
    for offset in offsets:
        marker_line[display_positions[offset]] = "|"
    lines.append("".join(marker_line).rstrip())

    descriptions_by_offset: dict[int, list[str]] = {}
    for offset, description in sorted_descriptions:
        descriptions_by_offset.setdefault(offset, []).append(description)

    for index, offset in enumerate(reversed(offsets)):
        remaining_offsets = [remaining for remaining in offsets if remaining < offset]
        for description_index, description in enumerate(descriptions_by_offset[offset]):
            line = [" "] * display_width
            for remaining in remaining_offsets:
                line[display_positions[remaining]] = "|"

            connector = (
                "`"
                if description_index == len(descriptions_by_offset[offset]) - 1
                else "|"
            )
            position = display_positions[offset]
            line[position] = connector
            line[position + 1 :] = ["-"] * (display_width - position - 1)
            lines.append(f"{''.join(line)} {description}")

        if index != len(offsets) - 1:
            separator = [" "] * display_width
            for remaining in remaining_offsets:
                separator[display_positions[remaining]] = "|"
            lines.append("".join(separator).rstrip())

    return "\n".join(lines)
