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
- **State:** 1,168 tests passing (2026-10-12, Phase 1b first batch)
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
| Positions capped at **20%** of capital (stocks, futures), **10%** (options) | Risk-based sizing with a tight stop otherwise puts 91% of the account in one trade |
| Futures count their **exchange margin**, not their value, against that cap | Justin's call (2026-10-12): margin is set by the exchange, not borrowing. P&L stays on the whole lot. Margin %s are Zerodha's on 12 Oct 2026 (`nlt/data/futures.py`, `MARGIN_AS_OF`), used for every year and stated in the report |
| Default account **Rs 25,00,000** | Justin's call (2026-10-12). At Rs 1 lakh the 20% cap is less than one unit of NIFTY |
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
| `nlt/orders/` | One-off conditional orders (ATO-style), paper only: reader, readback, engine, checker |
| `app/orders_view.py` | The one-off orders page, server side |
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

### Phase 1b — Reading and reporting, from the competitor review — **first batch DONE (2026-10-12)**

On 2026-10-11/12 FinStocks, Tradara and Tradl were reviewed (Tradl logged in:
its AI was given our sentences, and its 26 backtest + 17 screen suggested
prompts were run through our reader). Justin agreed the order below. Item codes
(A1, C3...) are the wish-list numbers he prioritised by; raw notes are in
`~/tradl-review/`, outside the repo.

**First batch -- built 2026-10-12**, branch `phase-1b-first-batch`. Every rule
below was broken on purpose and a test was seen to fail; three tests that did
not were tightened.

- **B3** Tradl's 43 prompts are `tests/fixtures/template_prompts.json`, apart
  from the 70. Each complete strategy carries the facts a human read off it
  (symbol, bar size, stop, target, square-off): a plausible read with any of
  them different counts as wrong. AI replies recorded in
  `llm_template_replies.json` (`scripts/eval_llm.py --templates`).
  **Result: 6 of Tradl's complete strategies now run on the AI path (was 0),
  each readback checked by hand; the 5 without a stop reach the stop question;
  0 wrong on the 43 and the 70.** The rules path alone still reads none of the
  43 (they need "On X bars, buy when...").
- **A3** "Backtest the last 2 years / past 6 months / for 2024" becomes a test
  period beside the spec (never on it); the screen fills its date boxes, never
  before the 2020 history floor. "hourly", "60 minute", "daily bars" read.
  "Hold overnight" turns off the same-day square-off, and the readback now says
  "Hold overnight" on minute/hourly bars; with a square-off too, it is a question.
- **A1** Each readback line is labelled **You said / We worked out / Default**.
  "You said" only when the reader saw it (or the user answered our question)
  *and* every number on the line was typed -- "RSI(14)" from "rsi" is "we
  worked out". Unknown is "Default", never "You said"; a saved strategy shows
  no labels. To make this possible the bar size is now its own readback line
  ("Daily candles") and a futures contract's lot facts are their own line.
  Surfaced on the way: a sentence naming no instrument silently means NIFTY --
  now visibly "Default".
- **A2** Question buttons, nothing pre-selected: stop 0.5/1/2% (both readers),
  an optional target, overnight vs square-off, and which indicator "it" means.
  Questions only a new description can answer no longer show a text box whose
  answer was silently ignored (that was every question except the stop).
- **B1/B2** A "What to include" guide (stop loss marked required) and five new
  examples, each tested to read cleanly and take trades at Rs 25 lakh.
- **C1** Every backtest is run a second time with no charges; the report shows
  both side by side. With a fixed size the gap equals the charges to the rupee.
- Also (Justin's call): futures on margin and the Rs 25 lakh default (see the
  engine rules), and the readback no longer states "Stop trading for the day
  after losing Rs 2,000" / "Never risk more than Rs 5,000 on one trade" --
  **no engine ever enforced them**. They come back, enforced, with D2.

Noticed, not fixed (cheap, for next time): the results notes still leak engine
words -- "schedule time-of-day rules (no_entry_after/square_off) skipped" on
every daily run, and "end_of_data" as an exit reason; the cost dropdown defaults
to *options* charges even for an index strategy; strategy names are built from
the first words ("On Infy Hourly Bars Buy When").

**Second batch:**

- **A4 Editable form before running** -- stop, target, chart, period as boxes,
  then "Run backtest" (Tradl, FinStocks). The readback must still match it.
- **C2 The report lists its own weak spots** -- e.g. today's lot sizes, only
  companies that exist today, fills assumed adverse (Tradara, Tradl).
- **C3 Plain list of assumptions**, plus "same strategy + same data = same
  result" (Tradl).
- **C4 Download trades as CSV** (Tradl).
- **C5 Indicator warm-up** -- check we don't trade before indicators have enough
  history, and say so in the report (Tradl does). Check first; may exist.
- **B4 Ready-made strategy library** -- cards with return, win rate, worst drop,
  trade count; tap, edit, it becomes yours (FinStocks, Tradl).
- **A5 Forgive typos** ("stcoks", "oversld") (Tradara).
- **A6 More conditions** -- Supertrend, Stochastic, Donchian, high of the first
  15 minutes, a daily average on an intraday chart (Tradl's prompts need these).

**Where we are already ahead -- keep it that way.** Tradl's own backtest of
"Buy RELIANCE at RSI 30, 5% stop, 10% target" bought **1 share** of a Rs 1,00,000
account and reported "+0.1%"; charges were off by default (yet listed); there
was no buy-and-hold comparison; it did not insist on a stop loss; its NIFTY
futures backtest crashed on a missing lot size; and a raw code dict leaked into
its assumptions list. Our sizing, charges, benchmark and stop rule exist to
prevent exactly these.

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

**From the competitor review (F3):** 11 of Tradl's 26 suggested backtest prompts
are options -- 9:20 short straddle, straddle with re-entry, iron condor, iron
butterfly, bull put spread, ATM call/put buys with premium stops, an IV-percentile
filter. They are a ready-made acceptance list for this phase.

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

**Next, from the competitor review** (paper first, carried into live):

- **E1 Live activity feed** -- every decision with its reason: "RSI below 30 ·
  order sent", "results in 2 days · skipped" (Tradara).
- **E2 Step-by-step timeline** per trade: "3:30 PM trigger met -> next open order
  queued -> 9:15 filled" (FinStocks).
- **E3 One drawer for running strategies** -- P&L for each, Pause and Stop
  (FinStocks).
- **E4 Alerts instead of trades** -- "tell me when...", by email or web (Tradara).

### Futures: MCX and NSE index — **BUILT (2026-10-07)**

Backtest and paper trade NIFTY/BANKNIFTY futures and eight MCX contracts: Crude
Oil, Natural Gas, Gold, Silver and their minis. Options on MCX wait with the
rest of Phase 2. Justin's decisions, each with its trade-off explained to him:

- **At expiry, close; never roll.** A position is closed at the expiry day's
  close and charged for that exit; the strategy re-enters only on a new signal.
  A signal on an expiry day's last candle is skipped (it would fill in the next
  contract on the old contract's levels). The benchmark follows the same rule.
- **Held on margin** (changed 2026-10-12; was "paid in full, no borrowing").
  A lot ties up its exchange margin -- NIFTY 11.32%, crude 31.25%, gold
  9.25% -- in the 20% cap, in spare capital and in "account too small". P&L is
  on the full lot. Every futures result says what margin was assumed and what a
  1% move means for it, and flags a stop wider than the margin. The futures
  benchmark still pays in full: it is "what the market did", not a leveraged bet.
  **Refresh the margin table** now and then; it goes stale with volatility.
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

Resolved 2026-10-12: the default account is now Rs 25 lakh, and the five
example strategies are each tested to take trades at it.

### One-off conditional orders (ATO-style) — **BUILT (2026-10-08), paper only**

Justin asked for "simple strategies" that work like Zerodha's Alert Trigger
Orders: *when this happens, do that, once.* His three examples, and what they
read as:

| Sentence | Result |
| --- | --- |
| If Nifty spot goes below 22000, buy 1 lot Nifty Futures | Order; readback warns there is no stop loss and offers to add one |
| Buy 1 lot Futures if Nifty has been going down for 4 consecutive weeks (4 consecutive red candles) | Order: 4 red weekly candles. Without the brackets, "going down" is a question: red candles, or lower closes? |
| If P&L has gone below Rs. 50000, exit positions | Question first: a loss of 50,000, or profit falling under 50,000? Then a list of open paper positions to tick |

**What real ATO can and cannot do** (researched 2026-10-08): one live value
(LTP, close, change %, OI, volume...) against a number or another instrument;
up to 20 orders fire once, then the alert disables itself. No indicators, no
candle patterns, no P&L condition (Zerodha staff say a P&L exit is coming "by
year end" 2026), no MCX/commodity ATO, 200 ATOs max, margin checked only when
it fires. Examples 2 and 3 are impossible as real ATOs; only example 1 is.

**Justin's decisions**, each given to him as a trade-off:

- **Paper only.** Nothing is sent to Zerodha. Real ATOs would mean the app
  placing orders, which is still undecided (see Phase 4).
- **No stop loss is allowed on a one-off order, but said out loud**: the
  readback says so, and the screen offers to add one. Strategies still always
  need one.
- **P&L exits look only at the app's own paper positions** (those opened by
  one-off orders), never the real Zerodha account.

**Rules**, the strategy engine's where they apply (`nlt/orders/engine.py`):
closed 1-minute candles only; a candle that began before the order was saved
never counts; fill at the next candle's open plus 0.02% slippage; stop first
when one candle hits both; a gap through the stop fills at the open; futures
closed at the expiry-day close, never rolled. Choices made for one-off orders,
written in the engine's docstring:

- Stops/targets are measured from the **fill price**, not a signal close --
  "1% stop" on a one-off order means 1% from what you paid.
- A new position never opens in a contract on its expiry day; it goes to the
  next month.
- "Below" means strictly below (Zerodha's `<`).
- P&L is **before charges**, like Kite's positions screen, over the chosen
  positions still open.
- A minute is judged one minute after it ends, and a daily/weekly candle an
  hour after the close, so a late-arriving candle is never skipped. The fill
  price is the same either way.
- A weekly run counts only weeks ending after the order is saved; if the run
  is already there, the screen warns that the next week must do the same.
- While the paper kill switch is on, minutes are **skipped**, not caught up on
  later at stale prices.

The sentence is re-read on save and must give the identical order, so what is
stored beside a sentence is always what it means. `one_off_event` is
append-only (SQLite triggers), like `paper_event`. The checker runs inside
`scripts/paper_run.py` every minute.

**Fixed on the way:** the paper cron line ran 08:30-16:29 IST only, so MCX
evenings (to 23:55) were never checked, for strategies either. It is now
08:30-00:29. **Re-run `scripts/install_paper_cron.sh`** to apply it.

`tests/test_one_off_orders.py`: each rule above was broken on purpose and a
test was seen to fail. None of the 70 user sentences becomes an order.

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

**Safety items from the competitor review**, to build with the risk chokepoint:

- **D1 Pre-order safety check** that can pass, shrink or refuse: results due
  within 2 days -> skip; order too large for the stock's daily volume -> refuse;
  India VIX above 25 -> halve the size. Thresholds in one readable table, not a
  model's opinion (Tradara).
- **D2 Hard daily limits** -- max automatic trades a day, max rupees per order,
  pause after 3 rejected orders in a row (Tradara: 50 / Rs 1,00,000 / 1 hour).
- **D3 Going live looks different** -- a separate, differently worded button:
  "This places real orders with real money" (FinStocks).
- **D4 Upfront risk statement** -- SEBI's finding that 9 in 10 individual F&O
  traders lost money, shown before anything goes live (FinStocks).

### Phase 5 — Deferred

Fundamentals (P/E, ROE, promoter holding — free via yfinance, verified working),
corporate events, multi-user, the factual-questions surface Justin chose
(answer verifiable questions, refuse predictions and recommendations).

**Added from the competitor review**, each a project in itself:

- **F1 Screen mode** -- "find oversold Nifty 50 stocks" returns a list. Tradl and
  Tradara both have it; several of the 70 sentences ask for it; Tradl's 17
  suggested screen prompts are an acceptance list.
- **F2 Many stocks in one strategy**, filtered and ranked (Tradara).
- **F4 Company-data and event strategies** -- P/E, ROE, promoter holding,
  dividend dates, buyback meetings, block deals (FinStocks, Tradara). Overlaps
  the fundamentals and corporate events above.
- **F5 News-based strategies** -- "when a US tariff on Indian exports is
  announced" (FinStocks).
- **Research and engagement, nice to have** (Tradl): G1 "since featured" -- how a
  suggestion moved since we showed it; G2 a public track record for any calls;
  G3 events calendar (would also feed D1); G4 sector strength view; G5 morning
  briefing; G6 stock pages; G7 community (not a build item).

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
