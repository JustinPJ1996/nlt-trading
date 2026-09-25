# Handoff

Everything needed to pick this up in a fresh conversation. Written 25 September
2026, after roughly four days of work.

---

## What this is

Someone who cannot code types a trading strategy in plain English. The system
shows them what it understood, tests it against real market history with real
Zerodha charges deducted, and tells them honestly whether it is any good.
Eventually it places the orders itself.

```
YOU TYPE     buy nifty when rsi cracks 30, target 2%, stop loss 1%

IT SHOWS     Buy NIFTY when RSI(14) crosses below 30
             Take profit at +2%, stop loss at -1%
             Risk 1% of capital per trade

IT REPORTS   This strategy lost money, and it also did far worse than
             holding NIFTY.   -3.8%  vs  +86.4%
```

- **Repo:** `~/trading-strategy-builder`, GitHub `JustinPJ1996/nlt-trading` (public)
- **State:** 701 tests passing, 23 commits, everything pushed
- **Run it:** `.venv/bin/python scripts/fetch_data.py` then `.venv/bin/streamlit run app/main.py`

### Who the user is

Justin (`justin@sensibull.com`) works at Sensibull but **does not code and is not
technical**. Explain in plain language. Offer product decisions as trade-offs,
not architecture. He is a sharp reviewer of *behaviour* — several of the worst
bugs found so far came from him asking a simple question about what something
said on screen.

**Target audience:** retail traders who know a strategy but not how to implement
it. He supplied 70 sentences in their voice; they are in
`tests/fixtures/user_strategies.txt` and drove most of the roadmap.

---

## The decisions that shape everything

**The LLM writes a validated recipe, never code.** A strategy is structured,
type-checked data (`nlt/spec/models.py`); one engine interprets it. Chosen
because this will eventually place real orders: a spec can be checked before it
runs, and look-ahead bias is structurally impossible when the engine controls
what a condition can see.

**There is no LLM at runtime at all.** Justin declined an API key, so the English
parser is a deterministic table of phrasing patterns (`nlt/translate/rules.py`).
It costs nothing, gives identical results every time, and is testable. An LLM
path can slot in behind the same interface later.

**Refuse rather than guess.** Across all 70 real user sentences: 3 produce a
spec, 67 are honestly refused, **0 are wrong**. That property is defended in
layers and asserted by `test_no_wrong_parses_among_the_70_user_strategies`.
Protect it above any coverage number.

**The readback is the entire safety story.** A non-technical user cannot audit a
spec but can read "Buy NIFTY when RSI(14) crosses below 30" and say "no, that's
wrong". It is generated mechanically from the spec, never by a model. It has
twice been caught stating things that were not true, and both were treated as
serious bugs.

**Never silently invent a stop loss.** A strategy without one returns a question.
This is why the original Phase 1 acceptance sentence no longer runs as written.

**History starts 2020.** Measured: 112 impossible single-session moves across
NIFTY 50 over full history, exactly 1 from 2020 onwards. Yahoo's Indian equity
data is unadjusted for corporate actions before ~2010, and a fake 90% crash gets
traded as a real one — profitable fiction, reported as confidently as anything
else.

---

## The engine's rules

Ask before changing any of these; each was a deliberate decision.

| Rule | Why |
| --- | --- |
| Signals judged on **closed** bars only | Never the bar still forming |
| Fills at the **next** bar's open, plus slippage | No acting on a price you could not have got |
| Stops and targets measured from the **signal bar's close** | Justin's call: the levels belong to the setup. Both from one number |
| Stop assumed first when a bar could hit both | OHLC cannot show intrabar order; take the pessimistic branch |
| A gap past the stop fills **at the open** | Stops do not protect you through a gap; gaps past the target modelled too, so the fix is not one-sided |
| Positions capped at **20%** of capital (stocks), **10%** (F&O) | Risk-based sizing with a tight stop otherwise puts 91% of the account in one trade |
| Non-options default to risking **1%** of capital per trade | "1 lot" of a stock is one share — Rs 640 on a Rs 1,00,000 account tests nothing |
| Lot sizes are **today's**, from one table | NIFTY 65, BANKNIFTY 30 (NSE circular 28-Nov-2025). Stated in the output because it makes old years unfaithful |
| Options under Rs 10,00,000 capital → **critical** flag | A 65-unit lot is indivisible; a small account cannot size proportionately |

---

## Layout

| Path | What |
| --- | --- |
| `nlt/spec/models.py` | The recipe format and every validator |
| `nlt/translate/rules.py` | English → spec, pattern table, no LLM |
| `nlt/translate/readback.py` | Spec → plain English, deterministic |
| `nlt/indicators/` | 43 indicators matching TradingView's Pine formulas |
| `nlt/engine/backtest.py` | Single-instrument engine |
| `nlt/engine/basket.py` | Many symbols, one shared account |
| `nlt/engine/loader.py` | Symbol → bars (index / stock / universe) |
| `nlt/costs/charges.py` | Zerodha brokerage, STT, GST, stamp, exchange |
| `nlt/report/` | Benchmark, verdict, 12 warning flags |
| `nlt/data/` | Sources, session calendar, universes, quality guard |
| `nlt/store/db.py` | Strategies, runs, audit log, kill switch, proving gate |
| `app/` | Streamlit dashboard, 4 pages |
| `nlt/options/`, `nlt/risk/`, `nlt/broker/` | **Empty. Phases 2-4.** |

---

## Phases

### Phase 1 — English to backtest — **DONE**

Spec models, parser, readback, indicators, engine, cost model, dashboard. Plus
the known-answer verification the plan asked for.

**Two caveats.** The literal acceptance sentence *"Buy NIFTY when RSI cracks 30,
exit at +2%"* returns a question, because it has no stop loss — deliberate, and
better than the original criterion. And a good deal of Phase 2/3 groundwork
landed early: stocks, universes, baskets, the session-aware intraday engine.

### Phase 2 — Options execution — **NOT STARTED**

`nlt/options/` is empty. Needs: strike selection (ATM default, configurable
offset), expiry choice (nearest weekly), premium modelling for backtests, lot
handling, mandatory square-off before expiry, and a plain-English record of why
each contract was chosen.

**Blocked on data.** Kite does not serve historical data for expired contracts.
Options: pay a vendor (TrueData, GDFL), or model premiums with Black-Scholes and
label results as estimates while the paper phase records real chains.

The charge model already handles options correctly, including the trap that an
exercised ITM option is charged STT on intrinsic value at the buy side.

### Phase 3 — Paper trading — **NOT STARTED**

Live websocket feed, a runner on a schedule, a chain recorder to build the
options history we lack, backtest-vs-paper drift reporting. **Needs Kite Connect
(Rs 2,000/month).**

The intraday engine work is already done and dormant: session calendar,
square-off, an invariant that raises if an intraday position ever spans two
trading days. Waiting on data, not code.

### Phase 4 — Live — **NOT STARTED**

`nlt/risk/` and `nlt/broker/` are empty. Needs: an order router, every order
through one risk chokepoint, reconciliation on restart, the daily Kite token
flow, a pre-open health check.

**Hard prerequisites.** Since 1 April 2026 Zerodha rejects API orders from any
non-whitelisted IP, so **this devbox cannot be the trading host** — a VPS with a
static IP is required. Every order must carry an Algo-ID. And the dashboard needs
authentication before it has a live broker session behind it; a public tunnel
URL with a kill switch on it is not acceptable.

The store already enforces the proving gate: live requires a completed backtest
*and* a completed paper run, or an explicit recorded override. Editing a strategy
creates a new version that starts unproven, so approval cannot be inherited by
rules that have changed.

### Phase 5 — Deferred

Fundamentals (P/E, ROE, promoter holding — free via yfinance, verified working),
corporate events, multi-user, the factual-questions surface Justin chose
(answer verifiable questions, refuse predictions and recommendations).

---

## What the 70 sentences say about coverage

| Class | Count | Status |
| --- | ---: | --- |
| Questions and advice | 31 | Out of scope by decision — answer factual only, refuse predictions |
| Stock universes | 8 | Engine ready; parser handles some |
| Fundamentals | 8 | Phase 5 |
| Corporate events | 7 | Phase 5 |
| Futures / MCX | 5 | Not started |
| Single stocks | 3 | Works |
| Multi-leg options | 2 | Phase 2 |
| Robo / portfolio | 3 | Out of scope |

Five stock sentences are blocked on VWAP, which is **deliberately refused on
daily bars** — it resets each session, so on daily data it collapses to that
bar's typical price, which is not what anyone means. Unblocked by intraday data,
i.e. by Kite.

---

## How to work on this

**Justin wants Sonnet subagents with detailed briefs, and their reports verified.**
He cannot review code himself, so verification is the only quality gate. Brief
them with the product context, the failure mode that matters, exact files to
create and not to touch (they run concurrently), and demand a self-critique.

**Then break their work on purpose.** Every claim in this document was checked by
mutating the code and confirming a test fails. It has found, among others:

- An engine reporting **-113%** on a strategy that could not lose more than its
  stop, because it bought 75 units of the index per "lot"
- Stops filling at prices that never traded when the market gapped
- `nlt/data/` missing from **every push** — a `.gitignore` pattern matching at
  any depth; `git status` said clean the whole time
- Every stock position capped at **two shares**
- The verdict naming NIFTY as the benchmark when it had measured a stock basket
- The readback promising a square-off the engine explicitly skips

None of these were caught by tests that already passed. Several were in work
reported as complete and correct.

**Mutation-test your own tests too.** A test that cannot fail is worse than no
test — it manufactures confidence. One in this repo asserted "either outcome is
fine"; it was deleted.

`tests/test_repo_hygiene.py` now guards the gitignore class of bug: every source
file tracked, no source path ignored, no unanchored directory pattern.

---

## Open items for Justin

1. **Kite Connect, Rs 2,000/month** — gates Phases 2, 3 and 4, plus intraday and
   the VWAP strategies. Nothing meaningful advances without it.
2. **A VPS with a static IP** — needed before any live order, not before then.
3. **Historical options data** — pay a vendor, or accept modelled estimates and
   let the paper phase accumulate real chains.
4. **More sentences in his users' voice** — the first 70 reshaped the product
   more than any architectural decision.

## Things that will bite you

- `data/` is gitignored: a fresh clone needs `scripts/fetch_data.py` first.
- Two GitHub accounts exist. Commits are authored as
  `58790250+JustinPJ1996@users.noreply.github.com` so they count on `JustinPJ1996`;
  `justin@sensibull.com` belongs to the unused `JustinJPJ` account.
- A tunnel may be running: `~/.local/state/devbox-tunnels/8501.pid`. Quick tunnel
  URLs are public and unauthenticated.
- `app/report_fallback.py` is dead code from a concurrency race and can go.
- The basket engine refuses intraday deliberately — it has no square-off.
