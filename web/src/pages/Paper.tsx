// Paper trading: a saved strategy run forward on live Kite prices with
// pretend money. Nothing on this page can place a real order -- the server
// side has no code that could.

import { useCallback, useEffect, useState, type FormEvent } from "react";
import { api, ApiError, type PaperRun, type PaperState } from "../api";
import { Callout, ErrorLine, Working, type Tone } from "../components/Callout";
import { Readback } from "../components/Readback";
import { parseRupees } from "../format";

const HEALTH_TONE: Record<string, Tone> = { success: "good", info: "info", warning: "warning", error: "critical" };
const KITE_TONE: Record<string, Tone> = { connected: "good", expired: "critical", not_connected: "warning", unreachable: "warning" };

function message(err: unknown, fallback: string) {
  return err instanceof ApiError ? err.message : fallback;
}

function KiteSection({ kite, reload }: { kite: PaperState["kite"]; reload: () => void }) {
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
  const [secret, setSecret] = useState("");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function act(fn: () => Promise<unknown>, fallback: string) {
    setBusy(true);
    setError(null);
    try {
      await fn();
      setUserId("");
      setPassword("");
      setSecret("");
      setToken("");
      reload();
    } catch (err) {
      setError(message(err, fallback));
    } finally {
      setBusy(false);
    }
  }

  const onLogin = (e: FormEvent) => {
    e.preventDefault();
    act(() => api.kiteLogin(userId, password, secret), "Could not save that login.");
  };
  const onToken = (e: FormEvent) => {
    e.preventDefault();
    act(() => api.kiteToken(token), "Could not save that token.");
  };

  return (
    <section className="section" aria-labelledby="kite">
      <h2 className="section-title" id="kite">
        Kite connection
      </h2>
      <Callout tone={KITE_TONE[kite.state] ?? "info"}>{kite.line}</Callout>
      <details className="panel" style={{ marginTop: 12 }} open={kite.state !== "connected"}>
        <summary>{kite.has_login ? "Change your Kite login" : "Save your Kite login"}</summary>
        <form onSubmit={onLogin} style={{ marginTop: 16 }}>
          <p className="hint" style={{ marginBottom: 16 }}>
            Saved on this computer only, readable only by you, and never shown again. The app uses it to
            log in to Kite by itself each morning. If Kite says the password is wrong, it stops trying, so
            that Zerodha does not lock your account.
          </p>
          <label className="field">
            <span className="label">Kite user ID</span>
            <input type="text" autoComplete="off" value={userId} onChange={(e) => setUserId(e.target.value)} required />
          </label>
          <label className="field">
            <span className="label">Kite password</span>
            <input type="password" autoComplete="off" value={password} onChange={(e) => setPassword(e.target.value)} required />
          </label>
          <label className="field">
            <span className="label">Authenticator secret</span>
            <input type="password" autoComplete="off" value={secret} onChange={(e) => setSecret(e.target.value)} required />
            <span className="hint">The long key you set up the authenticator with, not the 6-digit code.</span>
          </label>
          <div className="btn-row" style={{ marginTop: 16 }}>
            <button className="btn" type="submit" disabled={busy}>
              Save login
            </button>
            {kite.has_login && (
              <button type="button" className="btn quiet" disabled={busy} onClick={() => act(() => api.kiteForget(), "Could not forget the login.")}>
                Forget my Kite login
              </button>
            )}
          </div>
        </form>
        <details style={{ marginTop: 20 }}>
          <summary className="small">Or paste a token by hand</summary>
          <form onSubmit={onToken} style={{ marginTop: 12 }}>
            <label className="field">
              <span className="label">Kite token</span>
              <input type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} />
              <span className="hint">
                The token stops working each morning, so this is a daily step. It is only ever used to read
                prices.
              </span>
            </label>
            <button className="btn secondary" type="submit" disabled={busy || !token} style={{ marginTop: 12 }}>
              Save token
            </button>
          </form>
        </details>
        {error && (
          <div style={{ marginTop: 12 }}>
            <ErrorLine error={error} />
          </div>
        )}
      </details>
    </section>
  );
}

function RunCard({ run, reload }: { run: PaperRun; reload: () => void }) {
  const [busy, setBusy] = useState<"check" | "stop" | null>(null);
  const [sure, setSure] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function check() {
    setBusy("check");
    setError(null);
    try {
      setNote((await api.paperCheck(run.run_id)).message);
      reload();
    } catch (err) {
      setError(message(err, "Could not check now."));
    } finally {
      setBusy(null);
    }
  }

  async function stop() {
    setBusy("stop");
    setError(null);
    try {
      await api.paperStop(run.run_id);
      reload();
    } catch (err) {
      setError(message(err, "Could not stop."));
      setBusy(null);
    }
  }

  return (
    <article className="panel">
      <div className="run-head">
        <div>
          <h3 className="item-name">
            {run.name} <span className="item-meta">version {run.version}</span>
          </h3>
          <p className="muted small">
            Started {run.started} with {run.capital_text}. Last checked {run.checked || "not yet"}.
          </p>
        </div>
        <span className={`tag ${HEALTH_TONE[run.health] === "good" ? "good" : HEALTH_TONE[run.health] === "critical" ? "critical" : HEALTH_TONE[run.health] === "warning" ? "warning" : ""}`}>
          {run.health_label}
        </span>
      </div>
      {run.message && <p style={{ marginTop: 8 }}>{run.message}</p>}

      {run.snapshot && (
        <div className="stats">
          <div className="stat">
            <div className="muted small">Account value</div>
            <div className="stat-value">{run.snapshot.equity}</div>
          </div>
          <div className="stat">
            <div className="muted small">Closed trades, profit or loss</div>
            <div className="stat-value">{run.snapshot.realised}</div>
          </div>
          <div className="stat">
            <div className="muted small">Open position if closed now</div>
            <div className="stat-value">{run.snapshot.open}</div>
          </div>
        </div>
      )}

      {run.positions.length > 0 && (
        <div className="table-wrap" style={{ marginBottom: 16 }}>
          <table>
            <thead>
              <tr>
                <th scope="col">Open position</th>
                <th scope="col" className="num">
                  Entered at
                </th>
                <th scope="col" className="num">
                  Last price
                </th>
                <th scope="col" className="num">
                  If closed now, after costs
                </th>
              </tr>
            </thead>
            <tbody>
              {run.positions.map((p, i) => (
                <tr key={i}>
                  <td>
                    {p.direction} {p.what}
                  </td>
                  <td className="num">{p.entry_price}</td>
                  <td className="num">{p.last_price}</td>
                  <td className={`num ${p.pnl && p.pnl > 0 ? "pos" : p.pnl && p.pnl < 0 ? "neg" : ""}`}>{p.pnl_text}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <details>
        <summary>What the strategy is</summary>
        <div style={{ marginTop: 12 }}>
          <Readback text={run.readback} compact />
        </div>
      </details>
      <details style={{ marginTop: 12 }} open={run.events.length > 0}>
        <summary>Activity ({run.events.length})</summary>
        {run.events.length === 0 ? (
          <p className="muted" style={{ marginTop: 8 }}>
            Nothing has happened yet.
          </p>
        ) : (
          <ul className="log" style={{ marginTop: 8 }}>
            {run.events.map((e, i) => (
              <li key={i}>
                <time>{e.when}</time>
                <span className={e.drift ? "drift" : undefined}>{e.text}</span>
              </li>
            ))}
          </ul>
        )}
      </details>

      <div className="btn-row" style={{ marginTop: 20, justifyContent: "space-between" }}>
        <div className="btn-row">
          <button className="btn secondary" onClick={check} disabled={busy !== null}>
            Check now
          </button>
          {busy === "check" && <Working>Checking live prices...</Working>}
          {note && !busy && <span className="muted small">{note}</span>}
        </div>
        <div className="btn-row">
          <label className="check small">
            <input type="checkbox" checked={sure} onChange={(e) => setSure(e.target.checked)} />
            <span>Stop this paper run</span>
          </label>
          <button className="btn danger" onClick={stop} disabled={!sure || busy !== null}>
            Stop
          </button>
        </div>
      </div>
      {run.daily && (
        <p className="hint">Daily strategy: it updates once a day, shortly after the 3:30 PM close.</p>
      )}
      {error && (
        <div style={{ marginTop: 12 }}>
          <ErrorLine error={error} />
        </div>
      )}
    </article>
  );
}

function StartForm({ state, reload, onBuild }: { state: PaperState; reload: () => void; onBuild: () => void }) {
  const [pick, setPick] = useState<number | null>(state.startable[0]?.id ?? null);
  const [capital, setCapital] = useState(String(state.default_capital));
  const [cost, setCost] = useState(state.cost_models[0]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (state.startable.length === 0) {
    return (
      <div className="panel empty">
        <h2>Nothing to start</h2>
        <p className="muted" style={{ marginBottom: 20 }}>
          Save a strategy from its backtest results first, then start it here.
        </p>
        <button className="btn secondary" onClick={onBuild}>
          Go to my strategies
        </button>
      </div>
    );
  }

  const chosen = state.startable.find((s) => s.id === pick) ?? state.startable[0];
  const money = parseRupees(capital);

  async function start(e: FormEvent) {
    e.preventDefault();
    if (money === null) return;
    setBusy(true);
    setError(null);
    try {
      await api.paperStart(chosen.id, money, cost);
      reload();
    } catch (err) {
      setError(message(err, "Could not start paper trading."));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="panel" onSubmit={start}>
      <label className="field">
        <span className="label">Strategy</span>
        <select value={chosen.id} onChange={(e) => setPick(Number(e.target.value))}>
          {state.startable.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name} (version {s.version})
            </option>
          ))}
        </select>
      </label>
      <div style={{ margin: "16px 0" }}>
        <Readback text={chosen.readback} compact />
      </div>
      {chosen.problem ? (
        <Callout tone="warning">{chosen.problem}</Callout>
      ) : (
        <>
          <div className="grid-2">
            <label className="field" style={{ marginTop: 0 }}>
              <span className="label">Pretend starting money</span>
              <span className="money">
                <span aria-hidden="true">Rs</span>
                <input
                  type="text"
                  inputMode="numeric"
                  value={capital ? Number(parseRupees(capital) ?? 0).toLocaleString("en-IN") : ""}
                  onChange={(e) => setCapital(e.target.value.replace(/[^0-9]/g, ""))}
                />
              </span>
            </label>
            <label className="field" style={{ marginTop: 0 }}>
              <span className="label">Trading costs to charge</span>
              <select value={cost} onChange={(e) => setCost(e.target.value)}>
                {state.cost_models.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>
          </div>
          <p className="hint" style={{ marginTop: 12 }}>
            No real orders are placed. From the moment you press Start, the app follows live Kite prices
            and records every decision as it happens.
          </p>
          <div className="btn-row" style={{ marginTop: 16 }}>
            <button className="btn" type="submit" disabled={busy || money === null || money < 1000}>
              Start paper trading
            </button>
            {busy && <Working>Starting...</Working>}
          </div>
        </>
      )}
      {error && (
        <div style={{ marginTop: 12 }}>
          <ErrorLine error={error} />
        </div>
      )}
    </form>
  );
}

export function Paper({ onBuild }: { onBuild: () => void }) {
  const [state, setState] = useState<PaperState | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(() => {
    api.paper().then(
      (s) => {
        setState(s);
        setError(null);
      },
      (err) => setError(message(err, "Could not load paper trading.")),
    );
  }, []);

  useEffect(reload, [reload]);

  return (
    <div className="narrow">
      <h1 className="page-title">Paper trading</h1>
      <p className="page-lede">
        Run a saved strategy forward on live prices with pretend money. Nothing here places a real order.
      </p>
      <ErrorLine error={error} />
      {!state && !error && <Working>Checking Kite and loading your runs...</Working>}
      {state && (
        <>
          <div className="stack">
            {state.kill_switch && (
              <Callout tone="critical" title="The paper kill switch is on">
                Every paper strategy is paused. Turn it off on the <a href="#/safety">Safety</a> page.
              </Callout>
            )}
            {!state.job_installed && (
              <Callout tone="warning" title="The background job is not installed">
                Paper strategies only update when you press "Check now".
              </Callout>
            )}
          </div>

          <KiteSection kite={state.kite} reload={reload} />

          <section className="section" aria-labelledby="running">
            <h2 className="section-title" id="running">
              Running now
            </h2>
            {state.runs.length === 0 ? (
              <p className="muted">Nothing is being paper traded yet.</p>
            ) : (
              <div className="stack">
                {state.runs.map((r) => (
                  <RunCard key={r.run_id} run={r} reload={reload} />
                ))}
              </div>
            )}
          </section>

          <section className="section" aria-labelledby="start">
            <h2 className="section-title" id="start">
              Start paper trading
            </h2>
            <StartForm key={state.startable.map((s) => s.id).join(",")} state={state} reload={reload} onBuild={onBuild} />
          </section>
        </>
      )}
    </div>
  );
}
