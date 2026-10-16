// One merged, newest-first feed of everything paper trading and one-off
// orders have recorded: every signal taken, every signal skipped and why,
// every fill and exit, every order step. Nothing here is written fresh --
// it is all `/api/activity`'s own text, built from `app/logic.py`'s describe
// functions over the recorded rows.

import { useCallback, useEffect, useState } from "react";
import { api, type ActivityItem } from "../api";
import { ErrorLine, Working } from "../components/Callout";
import { TradeTimeline, type TimelineTarget } from "../components/TradeTimeline";
import { message, when } from "../util";

export function Activity() {
  const [items, setItems] = useState<ActivityItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sourceId, setSourceId] = useState<number | "all">("all");
  const [skipsOnly, setSkipsOnly] = useState(false);
  const [open, setOpen] = useState<TimelineTarget | null>(null);

  const reload = useCallback(() => {
    api.activity({ skips_only: skipsOnly }).then(
      (r) => {
        setItems(r.items);
        setError(null);
      },
      (err) => setError(message(err, "Could not load the activity feed.")),
    );
  }, [skipsOnly]);

  useEffect(() => {
    reload();
    // Paper trading and one-off orders both change while you watch: refresh
    // every half minute, the same as the Orders page.
    const timer = window.setInterval(reload, 30_000);
    return () => window.clearInterval(timer);
  }, [reload]);

  const sources = (() => {
    const seen = new Map<string, { source: "strategy" | "order"; id: number; name: string }>();
    for (const it of items ?? []) {
      seen.set(`${it.source}:${it.source_id}`, { source: it.source, id: it.source_id, name: it.source_name });
    }
    return [...seen.values()];
  })();

  const shown = (items ?? []).filter((it) => sourceId === "all" || it.source_id === sourceId);

  function openItem(it: ActivityItem) {
    if (it.trade_key) setOpen({ kind: "trade", runId: it.source_id, tradeKey: it.trade_key });
    else if (it.source === "order") setOpen({ kind: "order", orderId: it.source_id });
  }

  return (
    <div className="narrow">
      <h1 className="page-title">Activity</h1>
      <p className="page-lede">
        Everything paper trading and one-off orders have recorded, newest first: every signal taken, every
        signal skipped and why, every fill and exit.
      </p>
      <ErrorLine error={error} />

      <div className="row spread">
        <label className="field mt-0 minw-field">
          <span className="label">Source</span>
          <select
            value={sourceId === "all" ? "all" : `${sourceId}`}
            onChange={(e) => setSourceId(e.target.value === "all" ? "all" : Number(e.target.value))}
          >
            <option value="all">Everything</option>
            {sources.map((s) => (
              <option key={`${s.source}:${s.id}`} value={s.id}>
                {s.source === "strategy" ? "Strategy: " : "Order: "}
                {s.name}
              </option>
            ))}
          </select>
        </label>
        <label className="check mt-0">
          <input type="checkbox" checked={skipsOnly} onChange={(e) => setSkipsOnly(e.target.checked)} />
          <span>Skipped signals only</span>
        </label>
      </div>

      {!items && !error && <Working>Loading the activity feed...</Working>}
      {items && shown.length === 0 && (
        <p className="muted mt-4">
          Nothing recorded yet.
        </p>
      )}
      {items && shown.length > 0 && (
        <ul className="log tall mt-4">
          {shown.map((it, i) => {
            const clickable = !!it.trade_key || it.source === "order";
            const content = (
              <>
                <time>{when(it.time)}</time>
                <span>
                  <strong className="muted small">{it.source_name}</strong>
                  {it.skip && <span className="tag ml-2">skipped</span>}
                  <br />
                  <span className={it.skip ? "skip" : it.kind === "drift" ? "drift" : undefined}>{it.text}</span>
                </span>
              </>
            );
            return (
              <li key={i} className={clickable ? "clickable" : undefined}>
                {clickable ? (
                  <button type="button" className="row-button" onClick={() => openItem(it)}>
                    {content}
                  </button>
                ) : (
                  content
                )}
              </li>
            );
          })}
        </ul>
      )}
      {open && <TradeTimeline target={open} onClose={() => setOpen(null)} />}
    </div>
  );
}
