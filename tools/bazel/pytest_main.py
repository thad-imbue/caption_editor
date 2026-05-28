"""Bazel py_test entry shim that defers to pytest.

py_test treats `srcs[0]` as the script to execute via `python <file>`.
That works for `unittest.main()` modules but does nothing for plain
pytest test files (which only define `test_*` functions and rely on
pytest's discovery to run them). This shim is the actual `srcs[0]`
for every test target in //transcribe; the real test file is passed
as a runtime arg via the rule's `args` attribute.

Usage from a BUILD rule:

    py_test(
        name = "foo_test",
        srcs = ["//tools/bazel:pytest_main.py", "foo_test.py"],
        main = "//tools/bazel:pytest_main.py",
        args = ["$(location :foo_test.py)"],
        ...
    )

The `transcribe_pytest` macro in //tools/bazel:pytest.bzl wraps this
boilerplate so individual test rules stay one-liners.
"""

import sys

import pytest


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            "pytest_main: no test file given. "
            "Expected `args = [\"$(location :foo_test.py)\"]` in the BUILD rule.\n"
        )
        return 2
    # Forward everything after argv[0] verbatim. Bazel's test runner may
    # append extra flags (e.g. `--test_filter=...` becomes pytest-style
    # selectors via TESTBRIDGE_TEST_ONLY which pytest's collection ignores
    # — that's fine for the first pass; we can wire `-k` later).
    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
