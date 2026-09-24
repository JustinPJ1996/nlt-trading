# Trading Strategy Builder

Describe a trading strategy in plain English. See what the system understood.
Test it against years of real NIFTY data, with Zerodha's actual charges taken
out. Decide whether it is worth trading.

```
YOU TYPE     buy nifty when rsi cracks 30, target 2%, stop loss 1%

IT UNDERSTANDS
             Buy NIFTY when RSI(14) crosses below 30
             Take profit at +2%, stop loss at -1%
             1 lot per trade

IT REPORTS   This strategy lost money, and it also did far worse than
             simply holding NIFTY.
             Strategy -3.4%   ·   Buy and hold +407%
```

---

## Running it

Two commands, from this folder.

```bash
.venv/bin/python scripts/fetch_data.py       # first time only, ~10 seconds
.venv/bin/streamlit run app/main.py
```

Then open the address it prints (usually `http://localhost:8501`).

If you are running this on a remote machine and want to reach it from your own
browser, ask Claude to start a tunnel — the dashboard is not reachable from
outside the machine by default.

### Nothing happens / it looks broken

- **"No module named nlt"** — you are in the wrong folder, or not using
  `.venv/bin/python`. Both commands above must be run from this directory.
- **No data / empty charts** — run the `fetch_data.py` line above.
- **Everything is stale** — the data refreshes itself, but only when the
  machine has a working internet connection. The Safety page shows how old
  the data actually is.

---

## What it can and cannot do

**It can**, today:

- Understand a strategy written in ordinary English, using any of 43 indicators
  (RSI, moving averages, MACD, Bollinger Bands, ADX, Supertrend, candlestick
  patterns, pivots, and so on).
- Ask you a question when your description is missing something important. It
  will never invent a stop loss you did not ask for.
- Refuse, out loud, when it does not understand. It does not guess.
- Backtest on NIFTY or BANKNIFTY daily bars back to 2007.
- Take out real Zerodha brokerage, STT, GST, stamp duty and exchange charges.
- Show you buy-and-hold beside your strategy, and warn you about the eleven
  common ways a backtest flatters itself.

**It cannot**, yet:

- Trade individual stocks, or screen a basket like "Nifty 100 stocks".
- Use fundamentals — P/E, ROE, debt, promoter holding.
- React to events — dividends, board meetings, block deals, news.
- Trade options or futures as instruments (the charge model knows them; the
  engine does not trade them yet).
- Place a real order. Nothing here touches a broker.

---

## The one number that matters

Every backtest shows your strategy **and** what simply buying NIFTY and holding
it would have done. That comparison is there because it is the easiest thing in
the world to be pleased by a strategy returning 20% and never ask what the
index did over the same years.

Alongside it are warning flags for the ways backtests mislead:

| Flag | What it is telling you |
| --- | --- |
| Too few trades | Not enough evidence to separate skill from luck |
| Driven by one trade | Remove that trade and the strategy looks completely different |
| Charges dominate | It only works before costs, which means it does not work |
| Severe drawdown | You would have had to sit through losing this much |
| Long losing streak | Most people abandon a strategy before it recovers |
| Rarely in market | The comparison to buy-and-hold is not apples to apples |

A green verdict means no critical problem was found. It does **not** mean the
strategy is good.

---

## How it is built, in one paragraph

Your sentence is turned into a **strategy recipe** — a structured, validated
description of the rules. The recipe is data, never code: nothing a language
model writes is ever executed. One engine reads the recipe and runs it. That
same engine is what will later run paper trading and live trading, so "it
worked in the backtest" refers to the same code path that will place orders.

The rules the engine follows:

- A signal is judged on a **closed** candle, never one still forming.
- The order fills at the **next** candle's open, plus slippage.
- Stops and targets are measured from the **signal candle's close**, so a 1%
  stop and a 2% target are two percentages of the same number.
- If a candle could have hit both your stop and your target, the **stop** is
  assumed to have happened first.
- If the market **gaps past your stop**, you are filled at the open, not at
  the stop. Stops do not protect you through a gap, and the backtest does not
  pretend otherwise.

---

## Layout

| Folder | What lives there |
| --- | --- |
| `nlt/translate/` | English in, strategy recipe out, plus the plain-English readback |
| `nlt/indicators/` | 43 indicators, matching TradingView's formulas |
| `nlt/spec/` | The recipe format and everything that validates it |
| `nlt/engine/` | The backtester |
| `nlt/costs/` | Zerodha brokerage, STT, GST, stamp duty, exchange charges |
| `nlt/report/` | Benchmark comparison, verdict and warning flags |
| `nlt/data/` | Market data and its cache |
| `nlt/store/` | Saved strategies, run history, audit log, kill switch |
| `app/` | The dashboard |
| `tests/` | The reason any of the above can be trusted |

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests are the point, not a formality. The most important one runs every
indicator twice — once over part of the data, once over all of it — and demands
the overlapping values be identical. If any indicator could see a future bar,
that test fails. Looking into the future is what makes a worthless strategy
look brilliant, and it is the single most common way a backtest lies.

## Before this ever trades real money

Not yet built, and deliberately so:

- **Static IP.** Since 1 April 2026 Zerodha rejects API orders from any
  non-whitelisted address. This machine cannot be the trading host.
- **Paper trading first.** A strategy cannot be armed live until it has a
  completed backtest and a completed paper run on record. Editing a strategy
  creates a new version that starts unproven again, so approval cannot be
  inherited by rules that have changed.
- **Authentication on the dashboard.** A public URL with a live broker session
  behind it is not acceptable, however convenient.
