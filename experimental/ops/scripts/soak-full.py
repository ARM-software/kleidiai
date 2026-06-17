#!/usr/bin/python3

#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import datetime
import logging
import os
import queue
import random
import re
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
time_id = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
SOAKDIR = f"soak-{time_id}"

# Test Plan
test_plan = set()
failed_tests = set()


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

        for future in self.futures.queue:
            # See queue_work() for explanation of this call.
            future.result()

        self.executor.shutdown()


def log_error(tm, cmdline, output):
    global LAST_ERROR

    with tm.lock:
        try:
            os.makedirs(SOAKDIR)
        except FileExistsError:
            pass

        filename = "{}/soak-error-{}.log".format(SOAKDIR, LAST_ERROR)
        with open(filename, "w") as f:
            f.write("Error log {}:\n".format(LAST_ERROR))
            f.write("Command line: {}\n".format(cmdline))
            f.write("Output:\n")
            f.write(output)

        LAST_ERROR += 1


def do_gemm_cmd(args):
    cmd = f"{BINARY} {DEFAULT_ARGS} {args}"
    logging.debug(f"do_gemm_cmd: \t{cmd}")

    sp = os.popen(cmd)
    output = sp.read()
    returncode = sp.close()

    if returncode is not None:
        if returncode == -512:
            raise KeyboardInterrupt

        return output + "Error/crash: return code = {}\n".format(returncode)

    return output


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


class testcase(object):
    def __repr__(self):
        if self.dump_file:
            return f"testcase(dump, file={self.dump_file})"
        elif self.args.run_test_plan:
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

        if self.args.run_test_plan:
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
            main_args = f"-O {self.dump_file} {self.extra_args}"
        elif self.args.run_test_plan:
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
        lines = do_gemm_cmd(
            "{} {} -q {}".format(self.main_args(), self.extra_args, kernel)
        ).split("\n")
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

        if output.find("Compare OK") > -1:
            if not brief:
                print(
                    "Kernel {}, subkernel {} passed OK with {}".format(
                        kernel, subkernel, main_args
                    )
                )

            if self.args.test_plan_file and not args.run_test_plan:
                test_plan.add(main_args)

        elif output.find("Winograd: Unsupported kernel size:") > -1:
            print(
                "Kernel {}, subkernel {} skipped (unsupported size).".format(
                    kernel, subkernel
                )
            )
        elif output.find("does not support convolution problems, aborting.") > -1:
            print(
                "Kernel {}, subkernel {} skipped (non-convolution kernel).".format(
                    kernel, subkernel
                )
            )
        elif output.find("kai_ops_wrapper:") > -1 or output.find("Depthwise:") > -1:
            print(
                "Kernel {}, subkernel {} skipped (bad size).".format(kernel, subkernel)
            )
        elif output.find("emory allocation error.") > -1:
            print(
                "Kernel {}, subkernel {}: Aborted (memory allocation error).".format(
                    kernel, subkernel
                )
            )
        elif (
            output.find(
                "Accumulated value too large - test data not reassociation safe"
            )
            > -1
        ):
            print(f"Kernel {kernel}, subkernel {subkernel} skipped (bad data).")
        else:
            print(
                "Kernel {}, subkernel {}: Unexpected output, created error log.".format(
                    kernel, subkernel
                )
            )
            log_error(tm, thecmdline, output)

            if self.args.test_plan_file:
                failed_tests.add(main_args)


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

    if args.run_test_plan:
        assert args.test_plan_file != None
        assert not args.nobatch
        assert not args.gemv

    assert args.skip_cases >= 0

    if args.skip_cases != 0:
        assert (
            args.run_test_plan
        ), "We can only skip the tests in a deterministic test plan"

    # Setup logging
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(levelname)s: %(message)s")

    BINARY = args.binary

    kerns = get_kern_list()

    tm = ThreadManager(args.threads, args.threads * 3)

    total_cases = 0

    if args.dumpdir:
        logging.info("Test Dump Mode")
        directory = Path(args.dumpdir)

        for file in directory.iterdir():
            if file.is_file:
                parts = file.name.split("-")
                if parts[1] == "fast":
                    args_base = "-f"
                else:
                    args_base = ""

                make_case(tm, [parts[0]], args, dump_file=file, extra_args=args_base)
                make_case(
                    tm, [parts[0]], args, dump_file=file, extra_args=f"-P {args_base}"
                )
    elif args.run_test_plan:
        logging.info("Test Plan Mode")

        with open(args.test_plan_file, "r") as f:
            lines = f.readlines()

        tests = [line.strip() for line in lines if line.strip()]
        logging.info(f"Number of Test Cases: {len(tests)}")

        # Skip & Limit
        tests = tests[args.skip_cases :]
        tests = tests[: args.case_limit]
        logging.info(f"Number of Test Cases After Skip & Limit: {len(tests)}")

        for test in tests:
            make_case(tm, kerns, args, extra_args=test)
    else:
        logging.info("Random Test Mode")

        while True:
            print("{} case(s) processed, {} error(s)".format(total_cases, LAST_ERROR))
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
    if args.test_plan_file and not args.run_test_plan:
        with open(args.test_plan_file, "w") as file:
            for test_case in test_plan.difference(failed_tests):
                file.write(f"{test_case}\n")

    print(f"Complete: errors={LAST_ERROR}")

    end = time.time()
    logging.info(f"Elapsed Time: {end - start:.4f} seconds")

    sys.exit(LAST_ERROR > 0)
