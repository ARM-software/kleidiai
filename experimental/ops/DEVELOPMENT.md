<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Development

Make sure to install the pre-commit hooks before making any changes, like so

```bash
pip install pre-commit
pre-commit install
```

When you make a commit, the pre-commit hooks will run and lint your code. If any of the hooks
fail, address the error messages, stage the modified files again, then try to commit the code
again.

If for some reason, you are sure that the only remaining errors produced by the pre-commit hooks
are not related to your changes or need to be addressed in a separate branch/PR, you can bypass
the hooks with the `--no-verify` flag:

```bash
git commit [...] --no-verify
```
