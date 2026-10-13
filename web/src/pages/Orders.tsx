// One-off orders: "when this happens, do that, once" -- like Zerodha's
// alert-triggered orders, but paper only. Nothing on this page can place a
// real order; the server side has no code that could.
//
// The same rules as the Build page:
//  - The readback is the server's text, shown line by line, unchanged.
//  - Editing the sentence hides the readback; it belongs to the words it was read from.
//  - "Save" sends back the order that was read back, and the server reads the
//    sentence again to check it still means exactly that.
//  - A suggested answer is a hint, never pre-filled.

import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { api, ApiError, type OrderPosition, type OrderReading, type OrdersState, type SavedOrder, type Slippage, ZERO_SLIPPAGE } from "../api";
import { Callout, ErrorLine, Working, type Tone } from "../components/Callout";
import { SlippageInput } from "../components/SlippageInput";
import { TradeTimeline } from "../components/TradeTimeline";

const EXAMPLES = [
  "If Nifty spot goes below 22000, buy 1 lot Nifty Futures",
  "Buy 1 lot Futures if Nifty has been going down for 4 consecutive weeks (4 consecutive red candles)",
  "If P&L has gone below Rs. 50000, exit positions",
];

const STATUS_TAG: Record<SavedOrder["status"], string> = {
  waiting: "",
  filling: "warning",
  open: "good",
  done: "",
  cancelled: "",
  alerted: "good",
};

const CHECKER_TONE: Record<string, Tone> = {
  ok: "good",
  idle: "info",
  waiting: "info",
  paused: "critical",
  token_expired: "critical",
  not_connected: "warning",
  login_failed: "critical",
  feed_down: "warning",
  error: "critical",
};

function message(err: unknown, fallback: string) {
  return err instanceof ApiError ? err.message : fallback;
}

function Lines({ lines }: { lines: string[] }) {
  return (
    <div className="readback" data-testid="order-readback">
      {lines.map((line, i) => (
        <p className={i === 0 ? "readback-title" : "readback-line"} key={i}>
          {line}
        </p>
      ))}
    </div>
  );
}

function NewOrder({ onSaved }: { onSaved: () => void }) {
  const [text, setText] = useState("");
  const [checkedText, setCheckedText] = useState<string | null>(null);
  const [reading, setReading] = useState<OrderReading | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [stop, setStop] = useState("");
  const [chosen, setChosen] = useState<number[]>([]);
  const [slippage, setSlippage] = useState<Slippage>(ZERO_SLIPPAGE);
  const [busy, setBusy] = useState<"read" | "save" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);

  const stale = checkedText !== null && text.trim() !== checkedText;
  const r = stale ? null : reading;

  async function read(sentence: string, given: Record<string, string>) {
    const clean = sentence.trim();
    if (!clean) return;
    setBusy("read");
    setError(null);
    setSaved(null);
    try {
      const got = await api.orderRead(clean, given);
      setReading(got);
      setCheckedText(clean);
      setAnswers(got.answers);
      setDraft({});
      setChosen([]);
    } catch (err) {
      setError(message(err, "Could not read that order."));
    } finally {
      setBusy(null);
    }
  }

  function onRead(e: FormEvent) {
    e.preventDefault();
    // A changed sentence starts its questions afresh; answers belong to the words they answered.
    read(text, text.trim() === checkedText ? answers : {});
  }

  function onAnswers(e: FormEvent) {
    e.preventDefault();
    read(text, { ...answers, ...draft });
  }

  function onAddStop(e: FormEvent) {
    e.preventDefault();
    read(text, { ...answers, "trade.stop_loss": stop });
  }

  async function save() {
    if (!r?.order) return;
    setBusy("save");
    setError(null);
    try {
      await api.orderCreate(checkedText!, answers, r.order, chosen, slippage);
      setSaved(checkedText);
      setText("");
      setReading(null);
      setCheckedText(null);
      setAnswers({});
      setStop("");
      setChosen([]);
      setSlippage(ZERO_SLIPPAGE);
      onSaved();
    } catch (err) {
      setError(message(err, "Could not save that order."));
    } finally {
      setBusy(null);
    }
  }

  const picks = r?.needs_positions ? r.positions : [];
  const canSave = !!r?.order && (!r.needs_positions || chosen.length > 0);

  return (
    <div className="stack">
      <form className="describe" onSubmit={onRead}>
        <label className="visually-hidden" htmlFor="order-text">
          Your order, in plain English
        </label>
        <textarea
          id="order-text"
          ref={textRef}
          rows={2}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) onRead(e);
          }}
          placeholder="e.g. If Nifty spot goes below 22000, buy 1 lot Nifty Futures"
        />
        {!text && (
          <div className="examples" aria-label="Examples">
            <span className="muted small" style={{ alignSelf: "center" }}>
              Or try:
            </span>
            {EXAMPLES.map((ex) => (
              <button
                type="button"
                className="example"
                key={ex}
                onClick={() => {
                  setText(ex);
                  read(ex, {});
                }}
              >
                {ex}
              </button>
            ))}
          </div>
        )}
        <div className="btn-row" style={{ marginTop: 16 }}>
          <button className="btn" type="submit" disabled={busy !== null || !text.trim()}>
            Read my order
          </button>
          {busy === "read" && <Working>Reading your order...</Working>}
        </div>
      </form>

      {saved && (
        <Callout tone="good" title="Saved">
          It is now waiting below. Nothing is sent to Zerodha.
        </Callout>
      )}
      <ErrorLine error={error} />
      {stale && <p className="muted">You changed your order. Read it again to see what I understand now.</p>}

      {!!r && !r.order && r.questions.length > 0 && (
        <form className="panel reveal" onSubmit={onAnswers}>
          <p style={{ fontWeight: 650, marginBottom: 16 }}>I need a little more detail first.</p>
          {r.questions.map((q) => (
            <div className="question" key={q.field + q.text}>
              <label className="field">
                <span className="label">{q.text}</span>
                <input
                  type="text"
                  value={draft[q.field] ?? ""}
                  onChange={(e) => setDraft({ ...draft, [q.field]: e.target.value })}
                />
                <span className="hint">
                  {q.why}
                  {q.suggestion ? ` For example: ${q.suggestion}.` : ""}
                </span>
              </label>
            </div>
          ))}
          <div className="btn-row" style={{ marginTop: 8 }}>
            <button
              className="btn"
              type="submit"
              disabled={busy !== null || !r.questions.some((q) => (draft[q.field] ?? "").trim())}
            >
              Use these answers
            </button>
            <span className="muted small">Or change your sentence above and read it again.</span>
          </div>
        </form>
      )}

      {!!r?.order && (
        <div className="stack reveal">
          <p className="muted">Read this carefully. If anything is not what you meant, change your sentence.</p>
          <Lines lines={r.readback} />
          {r.notes.map((n, i) => (
            <Callout key={`n${i}`} tone="info">
              {n}
            </Callout>
          ))}
          {r.warnings.map((w, i) => (
            <Callout key={`w${i}`} tone="warning">
              {w}
            </Callout>
          ))}

          {r.offer_stop && (
            <form className="panel" onSubmit={onAddStop}>
              <Callout tone="warning" title="This order has no stop loss">
                If the price keeps moving against the position, nothing closes it until you do, or the
                contract expires. You can add one now.
              </Callout>
              <label className="field">
                <span className="label">Stop loss, as a percentage of the price it is bought or sold at</span>
                <span className="money">
                  <input
                    type="text"
                    inputMode="decimal"
                    value={stop}
                    onChange={(e) => setStop(e.target.value.replace(/[^0-9.]/g, ""))}
                    aria-describedby="stop-hint"
                  />
                  <span aria-hidden="true">%</span>
                </span>
                <span className="hint" id="stop-hint">
                  For example 1. Leave it empty to keep the order without one.
                </span>
              </label>
              <button className="btn secondary" type="submit" disabled={busy !== null || !stop}>
                Add this stop loss
              </button>
            </form>
          )}

          {r.needs_positions && picks.length > 0 && (
            <fieldset className="panel">
              <legend className="label">Which positions should this close?</legend>
              {picks.map((p) => (
                <label className="check" key={p.id} style={{ marginTop: 8 }}>
                  <input
                    type="checkbox"
                    checked={chosen.includes(p.id)}
                    onChange={(e) =>
                      setChosen(e.target.checked ? [...chosen, p.id] : chosen.filter((x) => x !== p.id))
                    }
                  />
                  <span>
                    {p.name}, entered {p.entered} at {p.entry_price}
                    {p.pnl_text ? `, now ${p.pnl_text}` : ""}
                  </span>
                </label>
              ))}
            </fieldset>
          )}

          <SlippageInput value={slippage} onChange={setSlippage} idPrefix="order-slippage" />

          <div className="btn-row">
            <button className="btn" onClick={save} disabled={busy !== null || !canSave}>
              Yes, save this order
            </button>
            <button
              className="btn secondary"
              onClick={() => {
                textRef.current?.focus();
                textRef.current?.select();
              }}
            >
              No, change my sentence
            </button>
            {busy === "save" && <Working>Saving...</Working>}
          </div>
          {r.needs_positions && picks.length > 0 && chosen.length === 0 && (
            <p className="hint">Tick at least one position to save this order.</p>
          )}
        </div>
      )}

      {r && !r.order && r.questions.length === 0 && (
        <Callout tone="warning" title="I could not read an order from that">
          Try the shape "If something happens, buy / sell / exit".
        </Callout>
      )}
    </div>
  );
}

function Positions({ positions, reload }: { positions: OrderPosition[]; reload: () => void }) {
  const [confirm, setConfirm] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function close(id: number) {
    setError(null);
    try {
      await api.orderClose(id);
      setConfirm(null);
      reload();
    } catch (err) {
      setError(message(err, "Could not close that position."));
    }
  }

  if (positions.length === 0) return <p className="muted">No open paper positions.</p>;
  return (
    <>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Position</th>
              <th scope="col" className="num">
                Entered at
              </th>
              <th scope="col" className="num">
                Last price
              </th>
              <th scope="col" className="num">
                Stop / target
              </th>
              <th scope="col" className="num">
                Profit or loss, before charges
              </th>
              <th scope="col">
                <span className="visually-hidden">Close</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p) => (
              <tr key={p.id}>
                <td>
                  #{p.id} {p.what}
                  <div className="muted small">
                    Since {p.entered}. Expires {p.expiry}.
                  </div>
                </td>
                <td className="num">{p.entry_price}</td>
                <td className="num">
                  {p.last_price || "-"}
                  {p.last_at && <div className="muted small">{p.last_at}</div>}
                </td>
                <td className="num">
                  {p.stop || "none"} / {p.target || "none"}
                </td>
                <td className={`num ${p.pnl && p.pnl > 0 ? "pos" : p.pnl && p.pnl < 0 ? "neg" : ""}`}>
                  {p.pnl_text || "-"}
                </td>
                <td>
                  {confirm === p.id ? (
                    <div className="btn-row">
                      <button className="btn danger" onClick={() => close(p.id)}>
                        Close at the next price
                      </button>
                      <button className="btn quiet" onClick={() => setConfirm(null)}>
                        Keep
                      </button>
                    </div>
                  ) : (
                    <button className="btn secondary" onClick={() => setConfirm(p.id)}>
                      Close now
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ErrorLine error={error} />
    </>
  );
}

function OrderCard({ order, reload }: { order: SavedOrder; reload: () => void }) {
  const [sure, setSure] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showTimeline, setShowTimeline] = useState(false);

  async function cancel() {
    setError(null);
    try {
      await api.orderCancel(order.id);
      reload();
    } catch (err) {
      setError(message(err, "Could not cancel that order."));
    }
  }

  return (
    <article className="panel">
      <div className="run-head">
        <div>
          <h3 className="item-name">
            #{order.id} {order.sentence}
          </h3>
          <p className="muted small">Saved {order.saved}.</p>
        </div>
        <span className={`tag ${STATUS_TAG[order.status]}`}>{order.status_label}</span>
      </div>
      {order.result_text && (
        <p style={{ marginTop: 8 }}>
          Result after charges:{" "}
          <span className={order.result && order.result > 0 ? "pos" : order.result && order.result < 0 ? "neg" : ""}>
            {order.result_text}
          </span>
        </p>
      )}
      <details style={{ marginTop: 12 }} open={order.status === "waiting" || order.status === "open"}>
        <summary>What this order does</summary>
        <div style={{ marginTop: 12 }}>
          <Lines lines={order.readback} />
        </div>
      </details>
      {order.events.length > 0 && (
        <>
          <ul className="log clickable" style={{ marginTop: 12 }} onClick={() => setShowTimeline(true)}>
            {order.events.map((e, i) => (
              <li key={i} className="clickable">
                <time>{e.when}</time>
                <span>{e.text}</span>
              </li>
            ))}
          </ul>
          <p className="hint">Click the activity above for this order's step-by-step timeline.</p>
        </>
      )}
      {showTimeline && (
        <TradeTimeline target={{ kind: "order", orderId: order.id }} onClose={() => setShowTimeline(false)} />
      )}
      {order.can_cancel && (
        <div className="btn-row" style={{ marginTop: 16, justifyContent: "flex-end" }}>
          <label className="check small">
            <input type="checkbox" checked={sure} onChange={(e) => setSure(e.target.checked)} />
            <span>Cancel this order</span>
          </label>
          <button className="btn danger" onClick={cancel} disabled={!sure}>
            Cancel
          </button>
        </div>
      )}
      <ErrorLine error={error} />
    </article>
  );
}

export function Orders() {
  const [state, setState] = useState<OrdersState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const reload = useCallback(() => {
    api.orders().then(
      (s) => {
        setState(s);
        setError(null);
      },
      (err) => setError(message(err, "Could not load your orders.")),
    );
  }, []);

  useEffect(() => {
    reload();
    // Orders change while you watch: refresh every half minute.
    const timer = window.setInterval(reload, 30_000);
    return () => window.clearInterval(timer);
  }, [reload]);

  async function check() {
    setChecking(true);
    setNote(null);
    try {
      setNote((await api.ordersCheck()).message);
      reload();
    } catch (err) {
      setError(message(err, "Could not check now."));
    } finally {
      setChecking(false);
    }
  }

  const active = state?.orders.filter((o) => o.status === "waiting" || o.status === "filling" || o.status === "open") ?? [];
  const past = state?.orders.filter((o) => !active.includes(o)) ?? [];

  return (
    <div className="narrow">
      <h1 className="page-title">One-off orders</h1>
      <p className="page-lede">
        Tell the app what to wait for and what to do when it happens, once. Like Zerodha's alert-triggered
        orders, but with pretend money: nothing here is sent to Zerodha.
      </p>
      <ErrorLine error={error} />
      {!state && !error && <Working>Loading your orders...</Working>}
      {state && (
        <>
          <div className="stack">
            {state.kill_switch && (
              <Callout tone="critical" title="The paper kill switch is on">
                No order is checked or filled while it is on, and minutes that pass meanwhile are skipped, not
                caught up on later. Turn it off on the <a href="#/safety">Safety</a> page.
              </Callout>
            )}
            {!state.job_installed && (
              <Callout tone="warning" title="The background job is not installed">
                Orders are only checked when you press "Check now".
              </Callout>
            )}
            {state.checker.message && (
              <Callout tone={CHECKER_TONE[state.checker.health] ?? "info"}>
                {state.checker.message}
                {state.checker.checked ? ` Last checked ${state.checker.checked}.` : ""}
              </Callout>
            )}
          </div>

          <section className="section" aria-labelledby="new-order">
            <h2 className="section-title" id="new-order">
              New order
            </h2>
            <NewOrder onSaved={reload} />
          </section>

          <section className="section" aria-labelledby="positions">
            <h2 className="section-title" id="positions">
              Open positions
            </h2>
            <Positions positions={state.positions} reload={reload} />
          </section>

          <section className="section" aria-labelledby="active">
            <div className="btn-row" style={{ justifyContent: "space-between" }}>
              <h2 className="section-title" id="active">
                Waiting and open
              </h2>
              <div className="btn-row">
                {checking && <Working>Checking live prices...</Working>}
                {note && !checking && <span className="muted small">{note}</span>}
                <button className="btn secondary" onClick={check} disabled={checking}>
                  Check now
                </button>
              </div>
            </div>
            {active.length === 0 ? (
              <p className="muted">No orders waiting.</p>
            ) : (
              <div className="stack">
                {active.map((o) => (
                  <OrderCard key={o.id} order={o} reload={reload} />
                ))}
              </div>
            )}
          </section>

          {past.length > 0 && (
            <section className="section" aria-labelledby="past">
              <h2 className="section-title" id="past">
                Finished and cancelled
              </h2>
              <div className="stack">
                {past.map((o) => (
                  <OrderCard key={o.id} order={o} reload={reload} />
                ))}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
