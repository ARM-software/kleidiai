#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
"""Encodes one AArch64 instruction as a KAI_ASM_INST line using llvm-mc."""
import argparse
import re
import subprocess
import sys


def main() -> int:
    """Assembles the instruction and prints its encoding and original text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction", help="One quoted AArch64 instruction")
    args = parser.parse_args()
    instruction = args.instruction.strip()

    try:
        result = subprocess.run(
            [
                "llvm-mc",
                "-triple=aarch64",
                "-mattr=+sve2",
                "-mattr=+sme2",
                "-show-encoding",
            ],
            input=instruction + "\n",
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        print(f"Unable to run llvm-mc: {error}", file=sys.stderr)
        return 1

    if result.returncode != 0:
        print(result.stderr.strip() or "llvm-mc failed.", file=sys.stderr)
        return 1

    encodings = re.findall(r"//\s*encoding:\s*\[([^\]]*)\]", result.stdout)
    if len(encodings) != 1:
        print("Expected exactly one instruction encoding.", file=sys.stderr)
        return 1

    encoding = encodings[0].strip()
    if not re.fullmatch(r"0x[0-9a-fA-F]{2}(?:\s*,\s*0x[0-9a-fA-F]{2}){3}", encoding):
        print("Expected four fully resolved hexadecimal bytes.", file=sys.stderr)
        return 1

    instruction_bytes = bytes(int(byte, 16) for byte in encoding.split(","))
    word = int.from_bytes(instruction_bytes, byteorder="little")
    print(f"KAI_ASM_INST(0x{word:08x})  // {instruction}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
