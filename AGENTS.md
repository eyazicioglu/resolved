# Agent Instructions

Before modifying any code:
1. Read `CONVENTIONS.md` for coding rules and architecture preferences.
2. Read `docs/SESSION_LOG.md` (top 3-5 entries) for recent context.
3. Read `docs/KNOWN_ISSUES.md` to avoid known pitfalls.
4. Read the relevant spec in `docs/specs/` if one exists.

## Developing a feature
1. If a spec exists in `docs/specs/{FEATURE}.md`, read it — that's the contract.
2. If no spec exists, write one with: problem, solution, changes required, tests, edge cases. Get approval before coding.
3. Implement in order: core logic → tests → integration → UI (if applicable).
4. After coding, do all of the following:
   - append a session entry to `docs/SESSION_LOG.md` (newest first)
   - update `docs/KNOWN_ISSUES.md` if you encountered or resolved any gotchas
   - update `CONVENTIONS.md` if you changed the directory structure or added new conventions

## Fixing a bug
1. Read `docs/KNOWN_ISSUES.md` — the bug may already be documented.
2. Reproduce → trace the code path → identify root cause (not just symptom).
3. Implement the minimal fix + add a regression test.
4. Add the bug to the **Resolved** section of `docs/KNOWN_ISSUES.md`.
5. Append a session entry to `docs/SESSION_LOG.md`.

## Refactoring
1. No behavior changes — inputs and outputs must remain identical.
2. All existing tests must still pass.
3. Update `CONVENTIONS.md` if directory structure changed.
4. Append a session entry to `docs/SESSION_LOG.md`.
