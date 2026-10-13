// The header "Running" drawer: one list of everything live right now --
// every paper strategy run, every waiting one-off order, every open one-off
// position. The money figures and health labels are never computed here --
// they arrive ready-made from `GET /api/running` (see `app/api.py:_running`),
// which itself reuses `_paper_run_card`'s and `orders_view.open_positions`'s
// own maths rather than redoing it.

import { useEffect, useState } from "react";
import { api, ApiError, type RunningItem, type RunningState } from "../api";
import { ErrorLine, Working } from "./Callout";

const HEALTH_CLASS: Record<string, string> = {
  success: "good",
  warning: "warning",
  error: "critical",
  info: "",
};

function message(err: unknown, fallback: string) {
  return err instanceof ApiError ? err.message : fallback;
}

function moneyClass(value: number): string {
  if (value > 0) return "pos";
  if (value < 0) return "neg";
  return "";
}

function StrategyActions({ item, onChanged }: { item: RunningItem; onChanged: () => void }) {
  const [busy, setBusy] = useState<"pause" | "stop" | null>(null);
  const [sure, setSure] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function togglePause() {
    if (item.run_id === undefined) return;
    setBusy("pause");
    setError(null);
    try {
      if (item.paused) {
        await api.paperResume(item.run_id);
      } else {
        await api.paperPause(item.run_id);
      }
      onChanged();
    } catch (err) {
      setError(message(err, "Could not change the pause."));
    } finally {
      setBusy(null);
    }
  }

  async function stop() {
    if (item.run_id === undefined) return;
    setBusy("stop");
    setError(null);
    try {
      await api.paperStop(item.run_id);
      onChanged();
    } catch (err) {
      setError(message(err, "Could not stop."));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="running-actions">
      <div className="btn-row">
        <button className="btn secondary small" onClick={togglePause} disabled={busy !== null}>
          {item.paused ? "Resume" : "Pause"}
        </button>
        <label className="check small">
          <input type="checkbox" checked={sure} onChange={(e) => setSure(e.target.checked)} />
          <span>Yes, stop it</span>
        </label>
        <button className="btn danger small" onClick={stop} disabled={!sure || busy !== null}>
          Stop
        </button>
      </div>
      {busy && <Working>Working...</Working>}
      {error && <ErrorLine error={error} />}
    </div>
  );
}

function RunningRow({ item, onChanged }: { item: RunningItem; onChanged: () => void }) {
  return (
    <li className="running-item">
      <div className="running-item-head">
        <div>
          <div className="item-name">{item.name}</div>
          <div className="muted small">
            {item.kind === "strategy" && "Paper strategy"}
            {item.kind === "one_off_order" && (
              <>
                One-off order &middot; <a href="#/orders">see on Orders</a>
              </>
            )}
            {item.kind === "one_off_position" && (
              <>
                One-off position &middot; <a href="#/orders">see on Orders</a>
              </>
            )}
          </div>
        </div>
        <span className={`tag ${HEALTH_CLASS[item.health] ?? ""}`}>{item.health_label}</span>
      </div>
      {item.paused && item.kind === "strategy" && <p className="muted small">Paused</p>}
      <div className="running-money">
        <div>
          <div className="muted small">Closed today</div>
          <div className={moneyClass(item.closed_today_value)}>{item.closed_today}</div>
        </div>
        <div>
          <div className="muted small">Open now{item.open_now_as_of ? ` (as of ${item.open_now_as_of})` : ""}</div>
          <div className={moneyClass(item.open_now_value)}>{item.open_now}</div>
        </div>
        <div>
          <div className="muted small">Total</div>
          <div className={moneyClass(item.total_value)}>{item.total}</div>
        </div>
      </div>
      {item.kind === "strategy" && <StrategyActions item={item} onChanged={onChanged} />}
    </li>
  );
}

export function RunningDrawer({ onClose }: { onClose: () => void }) {
  const [state, setState] = useState<RunningState | null>(null);
  const [error, setError] = useState<string | null>(null);

  function load() {
    api.running().then(
      (s) => {
        setState(s);
        setError(null);
      },
      (err) => setError(message(err, "Could not load what is running.")),
    );
  }

  useEffect(() => {
    load();
    const id = window.setInterval(load, 30_000);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <aside
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-label="Everything running"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <h3 className="item-name">Running</h3>
          <button className="btn quiet" onClick={onClose} aria-label="Close">
            Close
          </button>
        </div>
        {!state && !error && <Working>Loading...</Working>}
        <ErrorLine error={error} />
        {state && state.items.length === 0 && (
          <p className="muted">Nothing is running right now.</p>
        )}
        {state && state.items.length > 0 && (
          <>
            <ul className="running-list">
              {state.items.map((item, i) => (
                <RunningRow key={i} item={item} onChanged={load} />
              ))}
            </ul>
            <div className="running-totals">
              <div className="muted small">{state.totals.label}</div>
              <div className="running-money">
                <div>
                  <div className="muted small">Closed today</div>
                  <div>{state.totals.closed_today}</div>
                </div>
                <div>
                  <div className="muted small">Open now</div>
                  <div>{state.totals.open_now}</div>
                </div>
                <div>
                  <div className="muted small">Total</div>
                  <div>{state.totals.total}</div>
                </div>
              </div>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}
