---
name: python-testing
description: Write and run Python unit tests (stdlib unittest) and verify they pass.
---

Follow this skill when the task involves writing or running Python tests.

1. Put tests in a file named `<module>_test.py` in the workspace, using only
   the standard library `unittest` module (no extra dependencies).
2. Cover the happy path, edge cases, and error cases. Name test methods with
   `test_` so `unittest` discovers them automatically.
3. Run the tests with:
   `python -m unittest <module>_test -v`
4. Read the output. If any test fails, read the traceback, fix the code (or
   the test if the test itself is wrong), and re-run until all tests pass.
5. Report which tests exist and their final status; never claim success
   without a green test run.
