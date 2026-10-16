// A vertical step list for one trade or one order: every line comes straight
// from the server's `steps` array (built only from recorded rows -- see
// `app/logic.py:trade_timeline` and `app/orders_view.py:order_timeline`).
// Nothing here writes any wording of its own.

import { useEffect, useState } from "react";
import { api } from "../api";
import { ErrorLine, Working } from "./Callout";
import { message, useDialog } from "../util";

export type TimelineTarget = { kind: "trade"; runId: number; tradeKey: string } | { kind: "order"; orderId: number };

export function TradeTimeline({ target, onClose }: { target: TimelineTarget; onClose: () => void }) {
  const [steps, setSteps] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const closeRef = useDialog(onClose);

  useEffect(() => {
    setSteps(null);
    setError(null);
    const call =
      target.kind === "trade" ? api.paperTrade(target.runId, target.tradeKey) : api.orderTimeline(target.orderId);
    call.then(
      (r) => setSteps(r.steps),
      (err) => setError(message(err, "Could not load this timeline.")),
    );
  }, [target]);

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label={target.kind === "trade" ? "Trade timeline" : "Order timeline"}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <h3 className="item-name">{target.kind === "trade" ? "This trade, step by step" : "This order, step by step"}</h3>
          <button className="btn quiet" ref={closeRef} onClick={onClose} aria-label="Close">
            Close
          </button>
        </div>
        {!steps && !error && <Working>Loading...</Working>}
        <ErrorLine error={error} />
        {steps && steps.length === 0 && <p className="muted">Nothing recorded for this yet.</p>}
        {steps && steps.length > 0 && (
          <ol className="timeline">
            {steps.map((s, i) => (
              <li key={i}>
                <span className="timeline-dot" aria-hidden="true" />
                <span>{s}</span>
              </li>
            ))}
          </ol>
        )}
      </div>
    </div>
  );
}
