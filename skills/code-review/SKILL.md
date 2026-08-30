---
name: code-review
description: Review code for correctness bugs, error handling gaps, and edge cases.
---

Follow this skill when the task is to review existing code.

1. Read the target file(s) first; do not review from memory or description.
2. Check in this order:
   - correctness: wrong logic, off-by-one errors, incorrect assumptions;
   - error handling: unchecked failures, missing timeouts, swallowed errors;
   - edge cases: empty input, boundary values, unusual-but-valid inputs;
   - clarity: misleading names, dead code, commented-out blocks.
3. For each issue found, report: file, approximate line/function, severity
   (high/medium/low), and a concrete fix.
4. If you change code, re-run whatever tests or commands exist to confirm
   nothing broke.
5. End with a short summary: N issues found, of which X high severity, Y
   fixed (if you fixed any).
