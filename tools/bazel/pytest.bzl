"""Macro that wraps py_test so individual pytest files don't have to
repeat the entry-shim boilerplate.

py_test runs `python srcs[0]` and exits 0 on success — which silently
no-ops on a plain pytest file (just `def test_*` functions, no
`__main__`). //tools/bazel:pytest_main.py is the actual entry point;
the real test file is forwarded as the first positional arg so
`pytest.main()` discovers and runs it.

Snapshot tests (syrupy) read & write `__snapshots__/` relative to the
test file. Bazel's runfiles tree symlinks the test file from the
source location, so the snapshot dir is already siblings on disk and
no extra wiring is needed — as long as `__snapshots__/**` is listed
in the rule's `data`.
"""

load("@rules_python//python:defs.bzl", "py_test")

def transcribe_pytest(name, test_file, deps, data = [], **kwargs):
    """A py_test that invokes pytest on `test_file`.

    Args:
      name: target name.
      test_file: the pytest module (e.g. "captions_json5_lib_test.py").
      deps: list of py deps (libraries + pip `requirement(...)` calls).
            Must include `requirement("pytest")` — not auto-added so
            the call site documents what each test actually needs.
      data: extra runtime files (conftest, fixtures, snapshots).
      **kwargs: forwarded to py_test (tags, timeout, env, ...).
    """
    py_test(
        name = name,
        srcs = [
            "//tools/bazel:pytest_main.py",
            test_file,
        ],
        main = "//tools/bazel:pytest_main.py",
        args = ["$(location :%s)" % test_file],
        deps = deps,
        data = data,
        **kwargs
    )
