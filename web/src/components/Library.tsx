// Ready-made strategies: each card is a real backtest, recomputed once a day.
//
// A card must never look better than its report, so the return of simply
// holding sits beside the strategy's -- as a number and as a line -- and a
// test that raised a serious warning says so on the card. The page also says
// how many strategies were tried to pick these, and that most failed. Tapping
// a card puts its sentence in the box, where it is read and confirmed like
// anything the user typed.

import { useEffect, useState } from "react";
import { api, type Library as LibraryData, type LibraryCard } from "../api";
import { message } from "../util";
import { ErrorLine, Working } from "./Callout";

// Strategy and holding, on one axis (% on the starting money). Colours are the
// two validated chart slots; the numbers beside the swatches are the legend.
function Sparkline({ curve }: { curve: { s: number[]; b: number[] } }) {
  const W = 240;
  const H = 64;
  const all = [...curve.s, ...curve.b, 0];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const span = hi - lo || 1;
  const x = (i: number, n: number) => (n <= 1 ? 0 : (i / (n - 1)) * W);
  const y = (v: number) => 4 + (1 - (v - lo) / span) * (H - 8);
  const path = (vals: number[]) =>
    vals
      .map((v, i) => `${i ? "L" : "M"}${x(i, vals.length).toFixed(1)},${y(v).toFixed(1)}`)
      .join("");
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
      <line className="spark-zero" x1={0} x2={W} y1={y(0)} y2={y(0)} />
      <path className="spark-b" d={path(curve.b)} />
      <path className="spark-s" d={path(curve.s)} />
    </svg>
  );
}

function Card({ c, onPick }: { c: LibraryCard; onPick: (sentence: string) => void }) {
  return (
    <button type="button" className="library-card" onClick={() => onPick(c.sentence)}>
      <span className="library-card-head">
        <span className="library-card-title">{c.title}</span>
        {c.serious_warning && <span className="library-warning">Serious warning signs</span>}
      </span>
      <span className="library-sentence">{c.sentence}</span>
      {c.curve && <Sparkline curve={c.curve} />}
      <span className="library-numbers">
        <span className="library-figure">
          <span className="key s1" aria-hidden="true" />
          <b>{c.return}</b> strategy
        </span>
        <span className="library-figure">
          <span className="key s2" aria-hidden="true" />
          <b>{c.holding_return}</b> {c.benchmark_name}
        </span>
      </span>
      <span className="library-facts">
        <span>{c.trades} trades</span>
        <span>{c.win_rate} made money</span>
        <span>Worst drop {c.worst_drop}</span>
      </span>
      <span className="library-period">
        {c.first} to {c.last}
      </span>
    </button>
  );
}

export function Library({ onPick }: { onPick: (sentence: string) => void }) {
  const [data, setData] = useState<LibraryData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .library()
      .then((d) => !cancelled && setData(d))
      .catch((err) => !cancelled && setError(message(err, "Could not load them.")));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="library" aria-labelledby="library-title">
      <h2 className="section-title" id="library-title">
        Or start from a ready-made strategy
      </h2>
      {data?.tried && (
        <p className="library-intro">
          We have tested {data.tried} simple strategies on past prices. Most lost money or did worse
          than simply holding. That is why you test an idea before you trade it.
        </p>
      )}
      {!data && !error && <Working>Testing them on the latest prices...</Working>}
      {error && <ErrorLine error={error} />}
      {data && (
        <>
          <div className="library-grid">
            {data.cards
              .filter((c) => !c.error)
              .map((c) => (
                <Card key={c.title} c={c} onPick={onPick} />
              ))}
          </div>
          <p className="hint mt-3">
            Tested to {data.tested_to} with {data.capital_text} and real charges. Tap one to read
            it, change it and make it yours. Past results are not a promise.
          </p>
        </>
      )}
    </section>
  );
}
