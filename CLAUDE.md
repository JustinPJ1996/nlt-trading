# Working on this repository

Read this first. `HANDOFF.md` holds the full state — what is built, what is not,
and the reasoning behind every engine rule. Read it before changing anything.

## Who you are writing for

Justin does not code and is not technical. He is a sharp reviewer of
*behaviour* — several of the worst bugs in this project were found by him
asking a simple question about what something said on screen.

- Explain in plain English. No jargon, no architecture terms.
- Offer product decisions as **trade-offs**, not designs.
- He cannot review code, so **verification is the only quality gate.** Saying
  a thing works is not evidence that it does.

## Never read `legacy/`

It is an abandoned prototype describing an approach this project deliberately
moved away from. It is excluded from ruff and from the test suite. Reading it
will teach you the wrong thing.

## The rules that are decisions, not accidents

Ask before changing any of these. Each is written up in `HANDOFF.md` with the
reasoning:

- Signals judged on **closed** bars only; fills at the **next** bar's open.
- Stops and targets measured from the **signal bar's close**.
- Stop assumed first when one bar could hit both — OHLC cannot show intrabar
  order, so take the pessimistic branch.
- **Refuse rather than guess.** Across the 70 real user sentences in
  `tests/fixtures/user_strategies.txt`: 3 parse, 67 are refused, **0 are
  wrong**. `test_no_wrong_parses_among_the_70_user_strategies` defends that.
  It matters more than any coverage number.
- **Never silently invent a stop loss.** A strategy without one returns a
  question.
- **The readback is the entire safety story.** A non-technical user cannot
  audit a spec, but can read "Buy NIFTY when RSI(14) crosses below 30" and say
  "no, that's wrong". It is generated mechanically from the spec, never by a
  model. It has twice been caught stating something untrue, and both times that
  was treated as the most serious class of bug in this project.

## How to verify your own work

**A passing test proves nothing until you have seen it fail.**

Every claim in `HANDOFF.md` was checked by mutating the code and confirming a
test noticed. That practice found an engine reporting -113% on a strategy that
could not lose more than its stop, stops filling at prices that never traded,
and a readback promising a square-off the engine skipped. None were caught by
tests that already passed. Several were in work reported as complete.

So: after writing a test, **break the code it covers on purpose and confirm the
test fails.** `pnpm`-style automation for this is `make mutation` (see below).

Two traps this repo has already hit:

- **Tests bound to moving data.** Anything reading real market history must be
  pinned — see `NIFTY_FIXTURE_CUTOFF` in `tests/conftest.py`. A test anchored
  on `dt.date.today()` passes Monday to Friday and fails at weekends.
- **Stale bytecode during verification.** If a mutation and its original are
  the same file size, Python may reuse the cached `.pyc` and your check will
  report nonsense. Clear `__pycache__` between runs — `make mutation` does.

## The checks

```
make check      # ruff lint + format check + the full test suite. Run before committing.
make fix        # ruff autofix + format write. Run this before reading errors.
make mutation   # mutation testing (slow, deliberate — not part of `make check`)
make secrets    # gitleaks over the whole history
make hooks      # install the git hooks; needed once per clone
```

`make check` runs automatically on commit once `make hooks` has been run. It is
bypassable with `git commit --no-verify`, so a green branch you did not watch go
green is not evidence.

## The F&O dataset

`~/fno-data/` holds ClickHouse Native exports (not CSVs) read via `chdb`. Two
conversions are mandatory and silently catastrophic if missed:

- **Prices are in paise** — divide by 100.
- **Timestamps are UTC** — convert to Asia/Kolkata. `nlt/data/source.py`
  *assumes* a naive timestamp is IST rather than rejecting it, so a timezone-less
  frame will be 5h30m wrong and still pass validation.

Coverage is index F&O only (NIFTY, BANKNIFTY, SENSEX), 2022-12 to 2025-12.
