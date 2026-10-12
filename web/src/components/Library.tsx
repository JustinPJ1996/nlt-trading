// Ready-made strategies: each card is a real backtest, recomputed once a day.
//
// A card must never look better than its report, so the return of simply
// holding sits beside the strategy's, and a test that raised a serious warning
// says so on the card. Tapping a card puts its sentence in the box, where it is
// read and confirmed like anything the user typed.

import { useEffect, useState } from "react";
import { api, ApiError, type Library as LibraryData } from "../api";
import { ErrorLine, Working } from "./Callout";

export function Library({ onPick }: { onPick: (sentence: string) => void }) {
  const [data, setData] = useState<LibraryData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .library()
      .then((d) => !cancelled && setData(d))
      .catch((err) => !cancelled && setError(err instanceof ApiError ? err.message : "Could not load them."));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="library" aria-labelledby="library-title">
      <h3 className="library-title" id="library-title">
        Ready-made strategies
      </h3>
      {!data && !error && <Working>Testing them on the latest prices...</Working>}
      {error && <ErrorLine error={error} />}
      {data && (
        <>
          <div className="library-grid">
            {data.cards
              .filter((c) => !c.error)
              .map((c) => (
                <button type="button" className="library-card" key={c.title} onClick={() => onPick(c.sentence)}>
                  <span className="library-card-title">{c.title}</span>
                  {c.serious_warning && <span className="library-warning">Serious warning signs</span>}
                  <span className="library-numbers">
                    <span>
                      <b>{c.return}</b> strategy
                    </span>
                    <span>
                      <b>{c.holding_return}</b> {c.benchmark_name}
                    </span>
                  </span>
                  <span className="library-small">
                    {c.trades} trades · {c.win_rate} made money · worst drop {c.worst_drop}
                  </span>
                  <span className="library-small muted">
                    {c.first} to {c.last}
                  </span>
                  <span className="library-sentence">{c.sentence}</span>
                </button>
              ))}
          </div>
          <p className="hint">
            Tested to {data.tested_to} with {data.capital_text} and real charges. Tap one to read it,
            change it and make it yours. Past results are not a promise.
          </p>
        </>
      )}
    </section>
  );
}
