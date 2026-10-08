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

**The rules read first; an AI model reads only what they could not.** The
deterministic phrase table (`nlt/translate/rules.py`) costs nothing, gives
identical results every time, and is testable -- but it builds a recipe for only
3 of the 70 real sentences. On 2026-09-30 Justin chose to add a second reader via
OpenRouter (`nlt/translate/llm.py`, model `anthropic/claude-sonnet-5.5`, key in
`~/.config/nlt/openrouter_key`, never in the repo). The rules of that reader:

- It is asked only when the rules did not *understand* the words. It never
  overrules a deliberate refusal (options, candle size, VWAP on daily bars,
  "NIFTY 50" index-or-basket, an ambiguous "it"), and it is not asked when the
  rules understood everything and just need a stop loss.
- Instrument, direction and timeframe come from the rules' own extractors; the
  model may only fill a gap, and a disagreement becomes a question.
- Every part of the recipe must quote the user's exact words. Code checks each
  quote is really in the sentence (a fabricated quote discards the whole reply),
  that every number came from the user -- exit numbers from their own quote, next
  to "stop"/"target" -- and that no meaningful word was left out.
- Refusal wording is ours; the model only picks a category. Risk, sizing and
  schedule defaults are the rules' defaults. The readback is unchanged.
- No key, no network or a garbage reply: the rules' answer stands. The test
  suite sets `NLT_LLM_DISABLED` so no test can reach the model.

`scripts/eval_llm.py` runs the 70 live and saves the raw replies to
`tests/fixtures/llm_replies.json`; `tests/test_llm.py` replays them offline so
the 0-wrong property is defended on the AI path by every `make check`.

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
| `nlt/translate/llm.py` | Second-pass AI reader via OpenRouter, and every check on its answers |
| `nlt/data/kite.py` | Read-only Kite candles via the web-session token; the only door to Zerodha for data |
| `nlt/data/futures.py` | Futures contracts: lot sizes, expiry rules, MCX hours |
| `nlt/data/kite_login.py` | Logs in to Kite by itself (password + authenticator) to get that token; logs in, nothing else |
| `nlt/paper/runner.py` | Paper trading: the engine re-run each pass, plus an append-only record |
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

### Phase 3 — Paper trading — **BUILT (prototype)**

**Data: the Kite web-session workaround, not Kite Connect.** On 2026-10-03
Justin chose, on Balajee's suggestion and after being told the trade-offs, to
read candles with the `enctoken` from a logged-in Kite web session rather than
pay for Kite Connect. The trade-offs, so nobody rediscovers them: it goes
against Zerodha's terms and can break without notice; the token is a key to a
**real** trading account; it expires daily and must be pasted again. The free
Kite Connect "Personal" plan does **not** include market data. The paid Connect
plan is now **Rs 500/month** (not Rs 2,000) and includes live and historical
data; switching to it means changing `_get` in `nlt/data/kite.py`, nothing else.

`nlt/data/kite.py` is the only code that talks to Zerodha. It can only read:
`_get` sends GET to an allow-list (historical candles, profile) and refuses
everything else before sending; `tests/test_kite.py` reads the source and fails
on any POST/PUT/DELETE call or order path. The token lives in
`~/.config/nlt/kite_enctoken` (mode 600, outside the public repo). The app does
not tell the user how to extract the token from the browser -- an automated
safety check blocked writing those instructions; Balajee can show Justin.

**Automatic daily login (2026-10-07).** Justin, after confirming with Balajee
and being told the trade-offs, chose to stop pasting the token and let the app
log in by itself, adapted from a script Balajee shared (kept at
`~/kite_api/kite_api.py`, outside the repo). `nlt/data/kite_login.py` replays
the website login -- user id + password, then a 6-digit code it computes from
the authenticator secret -- and saves the `enctoken`. `kite._get` calls it when
there is no token or Kite rejects the old one, at most once per call. The cost:
the password and authenticator secret, stored in `~/.config/nlt/kite_login.json`
(mode 600), are together the full keys to the real account and do not expire.

Because the paper job runs every minute and Zerodha locks accounts after
repeated failures, logging in never hammers: a rejected password stops all
attempts until the login is saved again; a rejected code is retried once, 15
minutes later, then stops; a network failure waits 15 minutes. State lives in
`~/.config/nlt/kite_login_state.json`. `tests/test_kite_login.py` covers this;
each rule was checked by breaking it and watching a test fail. The Kite web
login also posts, so it cannot live in `kite.py`, whose test forbids any POST.
Pasting a token by hand still works as a fallback.

Live order placement is deliberately **left open**: Justin does not want the
integration limited to read-only forever, but has not decided to trade through
it. Nothing places orders today.

**How paper trading works** (`nlt/paper/runner.py`): every pass re-runs the
*same* backtest engine on closed candles up to now, with `trade_from` set so
nothing before Start opens a position, and records each entry/exit the first
time it is seen in `paper_event` -- append-only, enforced by SQLite triggers. If
revised candles change the engine's answer, a 'drift' row is added; the
original stands. `scripts/paper_run.py` runs every minute from cron
(`scripts/install_paper_cron.sh`; the crontab is outside the home directory, so
re-run that after a devbox rebuild). Intraday backtests now load from Kite too.

The replay test (`tests/test_paper.py`) feeds past candles one at a time and
requires exactly the backtest's trades. It found a real bug on its first run:
mid-session the engine took the newest candle for the day's last and squared
off on every pass. Fixed with `final_session_in_progress`.

There are **two kill switches** (Justin's call): live and paper, independent.

Saving a strategy from its results now records the backtest as a completed run
-- before, nothing did, so the proving gate could never pass.

**Not built:** the option-chain recorder, and any order placement (Phase 4).

### Futures: MCX and NSE index — **BUILT (2026-10-07)**

Backtest and paper trade NIFTY/BANKNIFTY futures and eight MCX contracts: Crude
Oil, Natural Gas, Gold, Silver and their minis. Options on MCX wait with the
rest of Phase 2. Justin's decisions, each with its trade-off explained to him:

- **At expiry, close; never roll.** A position is closed at the expiry day's
  close and charged for that exit; the strategy re-enters only on a new signal.
  A signal on an expiry day's last candle is skipped (it would fill in the next
  contract on the old contract's levels). The benchmark follows the same rule.
- **No borrowing.** A futures position is paid for in full, like a share.
- **20% per position**, the share limit (futures had none; options are 10%).
  A small account gets an "ACCOUNT TOO SMALL FOR ONE LOT" warning naming the
  capital that would work: one Gold lot is about Rs 1.5 crore.

Traps, all tested: Kite says lot size 1 for every MCX contract -- it is the order
unit, not the size (crude is 100 barrels; gold 1 kg priced per 10 g, so x100).
Expiry dates are computed: Kite keeps no list of expired contracts, and the folk
rules ("crude expires on the 19th") are wrong -- MCX energy expires one US
business day before the NYMEX contract, so US holidays move it. The rules are
held to an answer key (`tests/fixtures/futures_expiry_key.json`): every Kite
listing on 2026-10-07 and 509 contract switches since 2020 seen in Kite's open
interest; three Diwali/Gurpurab-week bullion expiries are listed exceptions.
MCX closes 23:30, or 23:55 in winter; paper trading follows each market's hours.
Daily history is Kite's continuous series from 2020; **intraday futures history
is only the live contract's life** (Kite keeps none for expired contracts), and
after each expiry intraday paper runs never re-judge the old contract.

Also fixed on the way: a strategy with no trades was summarised as "made money,
but did worse than holding" at 0% vs 0%; a tie is no longer called worse.

### Web front end (TypeScript) — **BUILT (2026-10-07), not yet published**

Justin asked for a slick, easy UI in TypeScript over the unchanged Python
engine. `web/` (Vite + React, strict TypeScript) is layout only; `app/api.py`
(Starlette) serves it and a JSON API that calls `app/logic.py`. Run with
`make web` then `make serve` (127.0.0.1:8502). The Streamlit app still works and
is still the one behind the existing tunnel.

Rules it keeps, each tested in `tests/test_api.py` and each test seen to fail
when the rule was deliberately broken:

- **Same login** as Streamlit (login ID + password, `app/auth.py`), fail-closed,
  signed HttpOnly/SameSite=Strict/Secure cookie, 5 wrong tries locks an address
  out for 15 min, every write must be JSON. Every `/api/` route except login
  requires a session.
- **What is tested is what was confirmed.** The old screen re-read the sentence
  on "Run backtest"; the AI reader can answer differently twice. The API takes
  the confirmed spec back and runs it via `logic.backtest_spec`.
- **The readback is the engine's text.** The browser lays it out line by line;
  only the five known section headings are shown in sentence case. Editing the
  sentence hides the readback and withdraws the "Yes, that's right".
- A question's suggested answer is a hint, never pre-filled.
- `verdict.passed` is shown as "No serious warning signs", never a green
  "Passed": it only means no critical flag fired.
- The chart is downsampled to <=1200 points keeping each bucket's worst
  drawdown, so it agrees with the "biggest drop" figure.

Fixed on the way: answering the stop-loss question with "1%" (what its own hint
suggests) was silently ignored and the question came back; only "1" worked.

Open: with the default Rs 1,00,000 and NIFTY, the example strategies take no
trades (a lot exceeds the 20% position cap). Engine rule, so left for Justin.

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
| Futures / MCX | 5 | Futures built (2026-10-07); all 5 still refused, each for a specific missing rule |
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

1. **Kite Connect, Rs 500/month** — the official replacement for the web-session
   workaround Phase 3 uses today. Needed before anything is relied on, and for
   Phase 4 regardless.
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
