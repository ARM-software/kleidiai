#!/usr/bin/python3

#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import datetime
import json
import logging
import os
import queue
import random
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Default arguments
DEFAULT_ARGS = ""

# Command to run gemm
BINARY = "./gemm"

LAST_ERROR = 0
TESTS_RUN = 0
TESTS_PASSED = 0
time_id = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
SOAKDIR = f"soak-{time_id}"

# Test Plan
test_plan = set()
failed_tests = set()

SIDECAR_FILENAME = "soak-dumps.json"

# Arguments to `gemm` that consume the following argv token as their value.
OPTIONS_WITH_VALUE = {
    "-M",
    "-N",
    "-K",
    "-a",
    "-d",
    "-t",
    "-T",
    "-m",
    "-i",
    "-s",
    "-S",
    "-l",
    "-L",
    "-C",
    "-y",
    "-z",
    "-n",
    "-D",
    "-I",
    "-F",
    "-j",
    "-k",
    "-G",
    "-w",
    "-V",
    "-r",
    "-u",
    "-U",
    "-p",
    "-X",
    "-O",
    "-o",
    "--winograd-output-rows",
    "--winograd-output-cols",
    "--winograd-input-transform",
    "--winograd-weight-transform",
    "--winograd-output-transform",
    "--weight-format",
}

# Arguments whose values are already encoded in the binary dump.
STORED_DUMP_OPTIONS_WITH_VALUE = {"-M", "-N", "-K", "-a", "-y", "-z", "-D", "-n", "-p"}

# Flags already encoded in the binary dump.
# Keep -f out of this set: it affects dumped data, but also selects replay work.
STORED_DUMP_FLAGS = {"-b"}

# Options used only to query/create/read dumps, not replayed from sidecar data.
DUMP_CREATION_ONLY_OPTIONS_WITH_VALUE = {"-O", "-o", "-F"}
DUMP_CREATION_ONLY_FLAGS = {"-c", "-q"}


class ThreadManager:
    def __init__(self, maxthreads, queuesize):
        self.maxthreads = maxthreads
        self.queuesize = queuesize
        self.executor = ThreadPoolExecutor(max_workers=maxthreads)
        self.executor._work_queue = queue.Queue(maxsize=queuesize)
        self.lock = threading.Lock()
        self.futures = queue.Queue()

    def queue_work(self, func, args):
        logging.debug("Queueing work in the ThreadManager...")

        # Clean the futures log as we queue to raise errors early
        while not self.futures.empty() and self.futures.queue[0].done():
            future = self.futures.get()

            # This call raises exception if thread couldn't run for some reason.
            # This is especially necessary if there is a syntax issue
            # within the function being called. This ensures we return
            # a non-zero code to the caller.
            future.result()

        future = self.executor.submit(func, *args)
        self.futures.put(future)

    def shutdown(self):
        logging.debug("Shutting Down the ThreadManager...")

        self.wait()
        self.executor.shutdown()

    def wait(self):
        logging.debug("Waiting for ThreadManager work...")

        while not self.futures.empty():
            future = self.futures.get()
            # See queue_work() for explanation of this call.
            future.result()


def log_error(tm, cmdline, output):
    global LAST_ERROR

    with tm.lock:
        try:
            os.makedirs(SOAKDIR)
        except FileExistsError:
            pass

        filename = f"{SOAKDIR}/soak-error-{LAST_ERROR}.log"
        with open(filename, "w") as f:
            f.write(f"Error log {LAST_ERROR}:\n")
            f.write(f"Command line: {cmdline}\n")
            f.write("Output:\n")
            f.write(output)

        LAST_ERROR += 1


def record_test_result(tm, passed):
    global TESTS_RUN, TESTS_PASSED

    with tm.lock:
        TESTS_RUN += 1
        if passed:
            TESTS_PASSED += 1


def do_gemm_cmd(args):
    cmd = build_gemm_argv(args)
    logging.debug(f"do_gemm_cmd: \t{shlex.join(cmd)}")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr

    if proc.returncode != 0:
        if proc.returncode == -2:
            raise KeyboardInterrupt

        return output + f"Error/crash: return code = {proc.returncode}\n"

    return output


def build_gemm_argv(args):
    if isinstance(args, str):
        args = shlex.split(args)
    else:
        args = list(args)

    return shlex.split(BINARY) + shlex.split(DEFAULT_ARGS) + args


def gemm_cmdline(args):
    return shlex.join(build_gemm_argv(args))


def get_kern_list():
    lines = do_gemm_cmd("").split("\n")

    kerns = []

    state = "waiting"
    for l in lines:
        l = l.rstrip()
        if state == "waiting":
            if l == "List of available kernels:":
                state = "active"
        elif state == "active":
            if len(l) > 0 and l[0] == "\t":
                kerns.append(l[1:].split(" ")[0])
            else:
                return kerns

    raise Exception("Unable to parse kernel list.")


def load_test_plan(path, skip_cases=0, case_limit=None, with_indices=False):
    with open(path, "r") as f:
        lines = f.readlines()

    tests = [(index, line.strip()) for index, line in enumerate(lines) if line.strip()]
    tests = tests[skip_cases:]
    tests = tests[:case_limit]

    if with_indices:
        return tests

    return [test for _, test in tests]


def is_deterministic_plan_mode(args):
    return args.run_test_plan or args.create_dumpdir


def sanitize_filename_field(value):
    value = re.sub(r"[^A-Za-z0-9_.]+", "_", value).strip("_")
    return value or "case"


def option_value(tokens, option, default=None):
    try:
        index = tokens.index(option)
    except ValueError:
        return default

    if index + 1 >= len(tokens):
        return default

    return tokens[index + 1]


def dump_name_for_case(kernel, case_line, case_index):
    tokens = shlex.split(case_line)
    fast = "fast" if "-f" in tokens else "nonfast"
    bias = "accumulate" if "-A" in tokens else ("bias" if "-b" in tokens else "nobias")
    act = sanitize_filename_field(option_value(tokens, "-a", "noact"))

    m = option_value(tokens, "-M")
    n = option_value(tokens, "-N")
    k = option_value(tokens, "-K")
    test_id = option_value(tokens, "-n")

    if m is not None and n is not None and k is not None:
        shape = f"M{m}_N{n}_K{k}"
    elif test_id is not None:
        suite = option_value(tokens, "-D", "suite")
        shape = f"{sanitize_filename_field(suite)}_id{sanitize_filename_field(test_id)}"
    else:
        shape = "manual"

    fields = [
        kernel,
        fast,
        f"case{case_index:06d}",
        sanitize_filename_field(shape),
        bias,
        act,
    ]
    return "-".join(fields)


def replay_args_for_plan_line(case_line):
    tokens = shlex.split(case_line)
    replay_args = []

    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token in OPTIONS_WITH_VALUE:
            if index + 1 >= len(tokens):
                replay_args.append(token)
                index += 1
                continue

            value = tokens[index + 1]
            if (
                token not in STORED_DUMP_OPTIONS_WITH_VALUE
                and token not in DUMP_CREATION_ONLY_OPTIONS_WITH_VALUE
            ):
                replay_args.extend([token, value])
            index += 2
            continue

        if token.startswith("--") and "=" in token:
            option, value = token.split("=", 1)
            if (
                option in OPTIONS_WITH_VALUE
                and option not in STORED_DUMP_OPTIONS_WITH_VALUE
            ):
                replay_args.append(f"{option}={value}")
        elif token not in STORED_DUMP_FLAGS and token not in DUMP_CREATION_ONLY_FLAGS:
            replay_args.append(token)

        index += 1

    return replay_args


def dump_creation_args_for_plan_line(case_line):
    tokens = shlex.split(case_line)
    creation_args = []

    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token == "-t":
            index += 2 if index + 1 < len(tokens) else 1
            continue

        if token == "-P":
            index += 1
            continue

        creation_args.append(token)
        index += 1

    return shlex.join(creation_args)


def known_skip_reason(output):
    if output.find("Winograd: Unsupported kernel size:") > -1:
        return "unsupported size"
    if output.find("does not support convolution problems, aborting.") > -1:
        return "non-convolution kernel"
    if output.find("kai_ops_wrapper:") > -1 or output.find("Depthwise:") > -1:
        return "bad size"
    if output.find("emory allocation error.") > -1:
        return "memory allocation error"
    if (
        output.find("Accumulated value too large - test data not reassociation safe")
        > -1
    ):
        return "bad data"
    return None


def iter_dump_files(directory):
    for file in sorted(Path(directory).iterdir()):
        if file.name == SIDECAR_FILENAME:
            continue
        if file.name.startswith("."):
            continue
        if not file.is_file():
            continue
        yield file


def legacy_dump_entry_from_filename(path, known_kerns=None):
    parts = path.name.split("-")
    if len(parts) < 2:
        return []
    if known_kerns is not None and parts[0] not in known_kerns:
        return []

    args_base = "-f" if parts[1] == "fast" else ""
    return [
        (path, parts[0], args_base),
        (path, parts[0], f"-P {args_base}".strip()),
    ]


def sidecar_dump_entries(directory):
    sidecar = Path(directory) / SIDECAR_FILENAME
    with open(sidecar, "r") as f:
        metadata = json.load(f)

    entries = []
    for filename, dump_metadata in metadata.get("dumps", {}).items():
        path = Path(directory) / filename
        if not path.is_file():
            logging.warning("Skipping missing dump: %s", path)
            continue

        replay_args = dump_metadata.get("replay_args", [])
        if isinstance(replay_args, list):
            replay_args = shlex.join([str(arg) for arg in replay_args])

        entries.append((path, dump_metadata["kernel"], replay_args))

    return entries


def iter_dump_entries(directory, known_kerns=None):
    directory = Path(directory)
    if (directory / SIDECAR_FILENAME).is_file():
        return sidecar_dump_entries(directory)

    entries = []
    for file in iter_dump_files(directory):
        entries.extend(legacy_dump_entry_from_filename(file, known_kerns))

    return entries


def prepare_dump_directory(directory, overwrite=False):
    directory = Path(directory)

    if directory.exists() and not directory.is_dir():
        raise ValueError(f"{directory} exists and is not a directory")

    if directory.exists() and any(directory.iterdir()):
        if not overwrite:
            raise ValueError(
                f"{directory} already exists and is not empty; use --overwrite_dumpdir to replace it"
            )
        shutil.rmtree(directory)

    directory.mkdir(parents=True, exist_ok=True)
    return directory


class testcase(object):
    def __repr__(self):
        if self.dump_file:
            return f"testcase(dump, file={self.dump_file})"
        elif is_deterministic_plan_mode(self.args):
            return f"testcase(extra_args={self.extra_args})"
        elif self.conv:
            return f"testcase(conv, ID={self.ID}, type={self.conv_type}, bias={self.bias}, act={self.act})"
        else:
            return f"testcase(gemm, M={self.M}, N={self.N}, K={self.K}, batch={self.batches}, multi={self.multis}, bias={self.bias}, act={self.act})"

    def __init__(self, args, dump_file=None, extra_args=""):
        self.conv = args.conv
        self.dump_file = dump_file
        self.args = args
        self.extra_args = extra_args

        if self.dump_file:
            print(f"Initializing testcase from dump file: {dump_file} {extra_args}")
            return

        if is_deterministic_plan_mode(self.args):
            logging.debug("Running testcase from the test plan")
            return

        repeat = True

        self.threads = random.choice(["", "-t 2", "-t 4", "-t 7"])
        self.cpu = random.choice(
            [
                "",
                "-C 0x4111d050",
                "-C 0x4111d030",
                "-C 0x461f0010",
                "-C 0x410fd440",
                "-C 0x410fd460",
            ]
        )
        self.bias = random.choice(["", "-b", "-A"])
        self.act = random.choice(["", "-a relu", "-a relu6", "-a relu7"])
        self.fast = random.choice(["", "-f"])
        self.fixed_format = random.choice(["", "-P"])

        self.extra_args = f"{self.extra_args} {self.threads} {self.bias} {self.act} {self.fast} {self.fixed_format} {self.cpu}"

        if self.conv:
            self.ID = random.randrange(1, 2**31)
            self.conv_type = random.choice(
                ["-V convolution", "-V indirect", "-V im2row"]
            )

            print(
                f"Initializing testcase: ID={self.ID}, conv={self.conv_type}, {self.extra_args}"
            )
        else:
            while repeat:
                Krange = random.choice([(1, 33), (32, 200), (200, 2000)])
                Mrange = random.choice([(1, 4), (3, 100), (101, 2000)])
                Nrange = random.choice([(1, 33), (32, 201), (200, 2000)])

                self.M = random.randrange(Mrange[0], Mrange[1])
                if args.gemv:
                    self.M = 1
                self.N = random.randrange(Nrange[0], Nrange[1])
                self.K = random.randrange(Krange[0], Krange[1])
                self.multis = random.randrange(1, 3)
                self.batches = random.randrange(1, 3)
                self.pre = ""
                self.trans = ""

                if args.nobatch:
                    self.batches = 1

                # Don't allow really big test cases (>1B macs)
                totalmacs = self.M * self.N * self.K * self.multis * self.batches
                if totalmacs < 100000000:
                    repeat = False

            print(
                f"Initializing testcase: M={self.M}, N={self.N}, K={self.K}, multis={self.multis}, batches={self.batches} {self.extra_args}"
            )

    def main_args(self):
        if self.dump_file:
            main_args = f"-O {shlex.quote(str(self.dump_file))} {self.extra_args}"
        elif is_deterministic_plan_mode(self.args):
            main_args = f"{self.extra_args}"
        elif self.conv:
            main_args = (
                f"-D {self.args.suite} -n {self.ID} {self.conv_type} {self.extra_args}"
            )
        else:
            main_args = f"-M {self.M} -N {self.N} -K {self.K} -y {self.batches} -z {self.multis} {self.extra_args}"

        return re.sub(r"\s+", " ", main_args).strip()

    def get_subkernels(self, kernel):
        logging.debug("get_subkernels()")
        lines = do_gemm_cmd(f"{self.main_args()} -q {kernel}").split("\n")
        subkerns = []

        state = "waiting"
        for l in lines:
            l = l.rstrip()
            if state == "waiting":
                if l == "Available kernels:":
                    state = "active"
            elif state == "active":
                if len(l) > 0 and l[0] == "\t":
                    subkerns.append(l[2:].split(" ")[0])
                else:
                    break

        return subkerns

    def run(self, tm, kernel, subkernel, brief):
        logging.debug("run()")

        main_args = self.main_args()
        if subkernel is None:
            thecmdline = f"{main_args} -c {kernel}"
        else:
            thecmdline = f"{main_args} -F {subkernel} -c {kernel}"

        output = do_gemm_cmd(thecmdline)
        passed = output.find("Compare OK") > -1
        record_test_result(tm, passed)

        test_result_string = f"[{thecmdline}]"

        if passed:
            if not brief:
                print(f"Passed OK {test_result_string}")

            if (
                self.args.test_plan_file
                and not self.args.run_test_plan
                and not self.args.create_dumpdir
            ):
                test_plan.add(main_args)

        elif known_skip_reason(output):
            reason = known_skip_reason(output)
            if reason == "memory allocation error":
                print(f"ABORTED (memory allocation error): {test_result_string}")
            else:
                print(f"SKIPPED ({reason}): {test_result_string}")
        else:
            print(f"ERROR (logged): {test_result_string})")
            log_error(tm, thecmdline, output)

            if self.args.test_plan_file:
                failed_tests.add(test_result_string)


def run_one_kernel(tm, case, k, args):
    logging.debug(f"run_one_kernel() with {k}")
    subkerns = case.get_subkernels(k)
    logging.debug(f"Subkernels of {k}: {subkerns}")

    if len(subkerns) == 0 and args.legacy:
        case.run(tm, k, None, args.brief)

    for sk in subkerns:
        if sk.find(args.subkernel_filter) != -1:
            case.run(tm, k, sk, args.brief)


def make_case(tm, kerns, args, dump_file=None, extra_args=""):
    logging.debug("Making Case...")
    case = testcase(args, dump_file, extra_args)
    logging.debug(f"Case: {case}")

    for k in kerns:
        if k.find(args.kernel_filter) != -1:
            logging.debug(f"\tQueueing the case for kernel {k}")
            tm.queue_work(run_one_kernel, (tm, case, k, args))


def create_dump_for_case(tm, output_dir, case_line, case_index, kernel, args):
    creation_args = dump_creation_args_for_plan_line(case_line)
    case = testcase(args, extra_args=creation_args)
    subkerns = case.get_subkernels(kernel)
    logging.debug(f"Subkernels of {kernel}: {subkerns}")

    # We need to make sure at least one subkernel exists (i.e. that this is
    # not either a legacy kernel or unsupported on this platform), but we
    # don't need to force selection of any particular subkernel when we
    # create the dump.  This avoids issues where some kernels are listed on
    # the `-q` output but can't in fact support the presented problem
    # parameters (usually due to quantization parameters which aren't
    # available on the `-q` path).
    subkernel = next(
        (sk for sk in subkerns if sk.find(args.subkernel_filter) != -1), None
    )
    if subkernel is None:
        return {
            "status": "skipped",
            "reason": "no matching subkernel",
            "kernel": kernel,
            "case_index": case_index,
            "test_plan_line": case_line,
        }

    filename = dump_name_for_case(kernel, case_line, case_index)
    final_dump = output_dir / filename
    tmp_dump = output_dir / f".{filename}.tmp-{os.getpid()}-{threading.get_ident()}"
    tmp_dump.unlink(missing_ok=True)

    cmd_args = shlex.split(case.main_args()) + ["-c", "-o", str(tmp_dump), kernel]
    output = do_gemm_cmd(cmd_args)
    passed = output.find("Compare OK") > -1
    record_test_result(tm, passed)

    if passed and tmp_dump.is_file() and tmp_dump.stat().st_size > 0:
        os.replace(tmp_dump, final_dump)
        return {
            "status": "created",
            "filename": filename,
            "kernel": kernel,
            "case_index": case_index,
            "test_plan_line": case_line,
            "replay_args": replay_args_for_plan_line(case_line),
        }

    tmp_dump.unlink(missing_ok=True)
    reason = known_skip_reason(output)
    if reason is not None:
        return {
            "status": "skipped",
            "reason": reason,
            "kernel": kernel,
            "subkernel": subkernel,
            "case_index": case_index,
            "test_plan_line": case_line,
        }

    log_error(tm, gemm_cmdline(cmd_args), output)
    return {
        "status": "failed",
        "reason": "unexpected output",
        "kernel": kernel,
        "subkernel": subkernel,
        "case_index": case_index,
        "test_plan_line": case_line,
    }


def print_create_dump_result(result, brief):
    if result["status"] == "created":
        if not brief:
            print(
                f"Created dump {result['filename']} for kernel {result['kernel']}",
                flush=True,
            )
    elif result["status"] == "skipped":
        if not brief:
            print(
                f"Skipped dump for kernel {result['kernel']}, case {result['case_index']} ({result['reason']}).",
                flush=True,
            )
    else:
        print(
            f"Failed dump for kernel {result['kernel']}, case {result['case_index']} ({result['reason']}).",
            flush=True,
        )


def create_dumps_for_testcase_and_store(
    tm, results, output_dir, case_line, case_index, kernels, args
):
    for kernel in kernels:
        result = create_dump_for_case(
            tm, output_dir, case_line, case_index, kernel, args
        )
        with tm.lock:
            results.append(result)
            print_create_dump_result(result, args.brief)


def create_dumpdir(tm, kerns, args):
    output_dir = prepare_dump_directory(args.create_dumpdir, args.overwrite_dumpdir)
    tests = load_test_plan(
        args.test_plan_file,
        skip_cases=args.skip_cases,
        case_limit=args.case_limit,
        with_indices=True,
    )
    filtered_kerns = [k for k in kerns if k.find(args.kernel_filter) != -1]

    metadata = {
        "version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "binary": args.binary,
        "test_plan_file": args.test_plan_file,
        "kernel_filter": args.kernel_filter,
        "subkernel_filter": args.subkernel_filter,
        "default_args": DEFAULT_ARGS,
        "dumps": {},
        "skipped": [],
        "failed": [],
    }

    planned = len(tests) * len(filtered_kerns)
    created = 0
    skipped = 0
    failed = 0

    logging.info(f"Number of Test Cases: {len(tests)}")
    logging.info(f"Number of Filtered Kernels: {len(filtered_kerns)}")
    logging.info(f"Number of Planned Dumps: {planned}")

    results = []
    for case_index, case_line in tests:
        tm.queue_work(
            create_dumps_for_testcase_and_store,
            (tm, results, output_dir, case_line, case_index, filtered_kerns, args),
        )

    tm.wait()

    for result in sorted(results, key=lambda r: (r["case_index"], r["kernel"])):
        if result["status"] == "created":
            metadata["dumps"][result["filename"]] = {
                "test_plan_line": result["test_plan_line"],
                "case_index": result["case_index"],
                "kernel": result["kernel"],
                "replay_args": result["replay_args"],
            }
            created += 1
        elif result["status"] == "skipped":
            metadata["skipped"].append(result)
            skipped += 1
        else:
            metadata["failed"].append(result)
            failed += 1

    metadata["summary"] = {
        "planned": planned,
        "created": created,
        "skipped": skipped,
        "failed": failed,
    }

    sidecar = output_dir / SIDECAR_FILENAME
    with open(sidecar, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(
        f"Dump creation complete: planned={planned}, created={created}, skipped={skipped}, failed={failed}"
    )


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--brief",
        help="Don't print 'OK' results for individual kernels",
        action="store_true",
    )
    parser.add_argument("--binary", help="Set binary to run", default=BINARY)
    parser.add_argument("--threads", help="Set thread count", default=1, type=int)
    parser.add_argument(
        "--conv", help="Generate convolution testcases", action="store_true"
    )
    parser.add_argument("--kernel_filter", help="Set kernel filter", default="")
    parser.add_argument("--subkernel_filter", help="Set subkernel filter", default="")
    parser.add_argument("--gemv", help="Force GEMV cases (M==1)", action="store_true")
    parser.add_argument("--legacy", help="Run legacy kernels", action="store_true")
    parser.add_argument("--nobatch", help="Force batches==1", action="store_true")
    parser.add_argument(
        "--suite",
        help="Which suite to use (for convolution cases)",
        default="random_conv2d",
    )
    parser.add_argument(
        "--dumpdir", help="Directory of dumps to run", default=None, type=str
    )
    parser.add_argument(
        "--create_dumpdir",
        help="Create a directory of dumps from --test_plan_file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--overwrite_dumpdir",
        help="Replace an existing --create_dumpdir directory",
        action="store_true",
    )
    parser.add_argument(
        "--case_limit", help="Max test cases to generate/run", default=None, type=int
    )
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    parser.add_argument(
        "--test_plan_file",
        help="File that stores/will store the test configs",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--run_test_plan", help="Run a pre-determined set of tests", action="store_true"
    )
    parser.add_argument(
        "--skip_cases",
        help="Number of test cases to skip in the test plan",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    selected_modes = sum(
        [
            args.run_test_plan,
            args.dumpdir is not None,
            args.create_dumpdir is not None,
        ]
    )
    if selected_modes > 1:
        parser.error(
            "--run_test_plan, --dumpdir, and --create_dumpdir are mutually exclusive"
        )

    if args.run_test_plan:
        if args.test_plan_file is None:
            parser.error("--run_test_plan requires --test_plan_file")
        if args.nobatch:
            parser.error("--run_test_plan cannot be combined with --nobatch")
        if args.gemv:
            parser.error("--run_test_plan cannot be combined with --gemv")

    if args.create_dumpdir:
        if args.test_plan_file is None:
            parser.error("--create_dumpdir requires --test_plan_file")

    if args.overwrite_dumpdir and not args.create_dumpdir:
        parser.error("--overwrite_dumpdir requires --create_dumpdir")

    if args.skip_cases < 0:
        parser.error("--skip_cases must be non-negative")

    if args.skip_cases != 0 and not (args.run_test_plan or args.create_dumpdir):
        parser.error("We can only skip the tests in a deterministic test plan")

    # Setup logging
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(levelname)s: %(message)s")

    BINARY = args.binary

    kerns = get_kern_list()

    tm = ThreadManager(args.threads, args.threads * 3)

    total_cases = 0

    if args.dumpdir:
        logging.info("Test Dump Mode")
        for dump_file, kernel, extra_args in iter_dump_entries(args.dumpdir, kerns):
            make_case(tm, [kernel], args, dump_file=dump_file, extra_args=extra_args)
    elif args.create_dumpdir:
        logging.info("Create Dump Mode")
        try:
            create_dumpdir(tm, kerns, args)
        except ValueError as e:
            parser.error(str(e))
    elif args.run_test_plan:
        logging.info("Test Plan Mode")

        tests = load_test_plan(args.test_plan_file)
        logging.info(f"Number of Test Cases: {len(tests)}")

        # Skip & Limit
        tests = load_test_plan(
            args.test_plan_file,
            skip_cases=args.skip_cases,
            case_limit=args.case_limit,
        )
        logging.info(f"Number of Test Cases After Skip & Limit: {len(tests)}")

        for test in tests:
            make_case(tm, kerns, args, extra_args=test)
    else:
        logging.info("Random Test Mode")

        while True:
            print(f"{total_cases} case(s) processed, {LAST_ERROR} error(s)")
            total_cases += 1
            make_case(tm, kerns, args)
            if args.case_limit is not None and total_cases >= args.case_limit:
                break

    tm.shutdown()

    if failed_tests:
        logging.info("Failed Tests")
        for test in failed_tests:
            logging.info(f"\t{test}")

    # If a test plan file is given and we're not running from it,
    # save the test configurations to this file.
    if args.test_plan_file and not args.run_test_plan and not args.create_dumpdir:
        with open(args.test_plan_file, "w") as file:
            for test_case in test_plan.difference(failed_tests):
                file.write(f"{test_case}\n")

    print(
        f"Complete: errors={LAST_ERROR}, tests_run={TESTS_RUN}, tests_passed={TESTS_PASSED}"
    )

    end = time.time()
    logging.info(f"Elapsed Time: {end - start:.4f} seconds")

    sys.exit(LAST_ERROR > 0)
