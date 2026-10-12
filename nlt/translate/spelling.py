"""Forgiving typos (competitor review A5), within Justin's limits (2026-10-12).

Everyday words are fixed and the fix is said: "Read 'oversld' as 'oversold'".
A stock or index name is never fixed -- a wrong guess trades a different
company -- so a near miss after "buy"/"sell" becomes a question with one
button: "Did you mean RELIANCE?".

A word is only fixed when it is not a word the reader knows, not a symbol or
contract name, not a real English word (SCOWL's American list, kept in
`english_words.txt.gz`; its licence is beside it), and exactly one vocabulary
word is within one edit of it (two for words of eight letters or more). The
English check is what stops "bank 4% profit" being read as "band 4% profit".
A word typed in capitals is a name or an acronym ("ROCE") and is left alone,
and nothing is ever turned into a three-letter abbreviation ("roc"). A tie
fixes nothing: the reader then says it did not understand, as before. The vocabulary below is the reader's own -- strategy
words, not English at large -- so a fix can only ever turn an unknown word
into one the reader understands.
"""

from __future__ import annotations

import functools
import re

VOCABULARY = frozenset(
    """
    above below over under cross crosses crossed crossing crossover cracks crack dips dip
    drops drop falls fall rises rise rising falling climbs climb sinks touches touch breaks
    break breakout reaches reach goes average averages moving simple exponential weighted
    close closes closing open opens opening high highs highest low lows lowest upper lower
    middle band bands bollinger stochastic supertrend donchian ichimoku keltner momentum
    volume oversold overbought bullish bearish golden death engulfing hammer doji harami
    marubozu morning evening star shooting spinning hanging candle candles candlestick chart
    charts minute minutes hour hourly hours daily weekly week weeks day days month months
    year years backtest backtesting last past overnight hold holding square intraday buy
    sell short cover exit enter entry target stop loss profit take trailing trail risk
    capital percent trade trades lot lots quantity rupees stocks stock shares share index
    futures future option options strategy signal line period previous level levels green
    red consecutive times after before then when while between within only every first
    crude natural gold silver mini instantly price prices value turns turn trend strength
    divergence reversal support resistance pivot range rsi macd ema sma wma hma atr adx cci
    vwap obv mfi roc histogram channel channels gap gaps session sessions twice once
    """.split()
)


# What a typo may be fixed into: words, not three-letter abbreviations.
_FIXABLE = frozenset(w for w in VOCABULARY if len(w) >= 4)


def _distance(a: str, b: str) -> int:
    """Edits between two words, a swap of neighbours counting as one ("stcoks")."""
    rows = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        rows[i][0] = i
    for j in range(len(b) + 1):
        rows[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            rows[i][j] = min(rows[i - 1][j] + 1, rows[i][j - 1] + 1, rows[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                rows[i][j] = min(rows[i][j], rows[i - 2][j - 2] + 1)
    return rows[len(a)][len(b)]


def _allowed(word: str) -> int:
    return 2 if len(word) >= 8 else 1


def _unique_nearest(word: str, candidates) -> str | None:
    limit = _allowed(word)
    best: list[str] = []
    best_d = limit + 1
    for c in candidates:
        if abs(len(c) - len(word)) > limit:
            continue
        d = _distance(word, c)
        if d < best_d:
            best, best_d = [c], d
        elif d == best_d:
            best.append(c)
    return best[0] if len(best) == 1 and best_d <= limit else None


@functools.cache
def _symbols() -> frozenset[str]:
    """Every name that picks what is traded: stocks, indices, futures contracts."""
    from nlt.data.futures import FUTURES
    from nlt.translate.rules import _known_tickers

    names = {s.lower() for s in _known_tickers()} | {"nifty", "banknifty", "sensex"}
    return frozenset(names | {s.lower() for s in FUTURES})


@functools.cache
def _english() -> frozenset[str]:
    import gzip
    from pathlib import Path

    path = Path(__file__).with_name("english_words.txt.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return frozenset(line.strip() for line in f if line.strip())


def _known(word: str) -> bool:
    from nlt.translate.rules import _FILLER

    return word in VOCABULARY or word in _FILLER or word in _symbols() or word in _english()


_WORD = re.compile(r"(?<![a-z0-9_%])[a-z]{3,}(?![a-z0-9_])", re.IGNORECASE)


def fix_words(text: str) -> tuple[str, list[tuple[str, str]]]:
    """`text` with misspelt everyday words fixed, and each (typed, read as)."""
    fixes: list[tuple[str, str]] = []

    def one(m: re.Match) -> str:
        word = m.group().lower()
        if _known(word) or m.group().isupper():
            return m.group()
        nearest = _unique_nearest(word, _FIXABLE)
        if nearest is None or nearest in _symbols():
            return m.group()
        fixes.append((m.group(), nearest))
        return nearest

    return _WORD.sub(one, text), fixes


def symbol_near_miss(text: str) -> tuple[str, str] | None:
    """(the word typed, the symbol it is one slip from), for the word after the verb.

    Only that one place, the same place the reader looks for a stock name
    (`rules._find_stock_ticker`), and only when the word is not a known word.
    """
    from nlt.translate.rules import _DIRECTION_LEAD

    lowered = text.lower()
    lead = _DIRECTION_LEAD.match(lowered)
    if not lead:
        return None
    m = re.match(r"\s+(?P<word>[a-z][a-z0-9&]{2,19})\b", lowered[lead.end() :])
    if not m:
        return None
    word = m.group("word")
    if _known(word) or any(ch.isdigit() for ch in word):
        return None
    nearest = _unique_nearest(word, _symbols())
    return (word, nearest.upper()) if nearest else None


SPELLING_FIELD = "instrument.spelling"


def prepare(original: str, answers: dict[str, str]):
    """(text to read, notes, question or None) -- run before either reader.

    The spec keeps the user's words exactly as typed; only what is *read* is
    corrected, and every correction is named in a note.
    """
    from nlt.translate.rules import Choice, Question

    notes: list[str] = []
    text = original
    miss = symbol_near_miss(text)
    if miss is not None:
        typed, symbol = miss
        if answers.get(SPELLING_FIELD, "").strip().upper() == symbol:
            text = re.sub(rf"\b{re.escape(typed)}\b", symbol, text, count=1, flags=re.IGNORECASE)
            notes.append(f"Read '{typed}' as {symbol}, as you confirmed.")
        else:
            return (
                text,
                notes,
                Question(
                    text=f"Did you mean {symbol}?",
                    why=(
                        f"'{typed}' is not a symbol we know, and a guessed name would test a "
                        "different company. If it is not this one, write the NSE symbol."
                    ),
                    suggestion=None,
                    field=SPELLING_FIELD,
                    choices=[Choice(f"Yes, {symbol}", symbol)],
                ),
            )
    text, fixes = fix_words(text)
    notes.extend(f"Read '{typed}' as '{fixed}'." for typed, fixed in fixes)
    return text, notes, None
