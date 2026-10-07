import { useEffect, useState } from "react";
import { api, ApiError, type SafetyState } from "../api";
import { ErrorLine, Working } from "../components/Callout";

const SWITCHES = {
  live: {
    title: "Live trading",
    on: "On. Nothing can trade with real money.",
    off: "Off. Live trading is not blocked.",
    confirm: "I understand this stops every live strategy immediately.",
  },
  paper: {
    title: "Paper trading",
    on: "On. Every paper strategy is paused.",
    off: "Off. Paper strategies are running.",
    confirm: "I understand this pauses every paper strategy. Live trading is not affected.",
  },
} as const;

function KillSwitch({
  which,
  engaged,
  onChange,
}: {
  which: "live" | "paper";
  engaged: boolean;
  onChange: (s: SafetyState) => void;
}) {
  const [sure, setSure] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const text = SWITCHES[which];

  async function flip(action: "engage" | "release") {
    setBusy(true);
    setError(null);
    try {
      onChange(await api.killSwitch(which, action, action === "engage" ? sure : false));
      setSure(false);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not change the switch.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <div className="switch-row">
        <div>
          <h3 className="section-title" style={{ marginBottom: 6 }}>
            {text.title} kill switch
          </h3>
          <p className="switch-state">
            <span className="dot" style={{ background: engaged ? "var(--critical)" : "var(--good)" }} aria-hidden="true" />
            {engaged ? text.on : text.off}
          </p>
        </div>
        <div className="stack">
          {engaged ? (
            <button className="btn secondary" onClick={() => flip("release")} disabled={busy}>
              Turn off the kill switch
            </button>
          ) : (
            <>
              <label className="check">
                <input type="checkbox" checked={sure} onChange={(e) => setSure(e.target.checked)} />
                <span>{text.confirm}</span>
              </label>
              <button className="btn danger" onClick={() => flip("engage")} disabled={!sure || busy}>
                Turn on the kill switch
              </button>
            </>
          )}
        </div>
      </div>
      {error && (
        <div style={{ marginTop: 12 }}>
          <ErrorLine error={error} />
        </div>
      )}
    </div>
  );
}

export function Safety({ onChange }: { onChange: (s: SafetyState) => void }) {
  const [state, setState] = useState<SafetyState | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.safety().then(setState, (err) =>
      setError(err instanceof ApiError ? err.message : "Could not load the safety page."),
    );
  }, []);

  const update = (s: SafetyState) => {
    setState(s);
    onChange(s);
  };

  return (
    <div className="narrow">
      <h1 className="page-title">Safety</h1>
      <p className="page-lede">
        Two independent switches. Each stops one kind of trading at once, and stays on until you turn
        it off here.
      </p>
      <ErrorLine error={error} />
      {!state && !error && <Working>Loading...</Working>}
      {state && (
        <>
          <div className="stack">
            <KillSwitch which="live" engaged={state.live} onChange={update} />
            <KillSwitch which="paper" engaged={state.paper} onChange={update} />
          </div>

          <section className="section" aria-labelledby="fresh">
            <h2 className="section-title" id="fresh">
              How fresh the price history is
            </h2>
            <div className="table-wrap">
              <table>
                <tbody>
                  {state.freshness.map((f) => (
                    <tr key={f.symbol}>
                      <th scope="row" style={{ width: 140 }}>
                        {f.symbol}
                      </th>
                      <td>
                        <span className={`tag ${f.level}`}>
                          <span aria-hidden="true">{f.level === "good" ? "✓" : f.level === "critical" ? "!" : "i"}</span>
                          {f.level === "good" ? "Fresh" : f.level === "critical" ? "Old" : "None"}
                        </span>{" "}
                        <span className="muted">{f.text}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="section" aria-labelledby="audit">
            <h2 className="section-title" id="audit">
              Recent activity
            </h2>
            {state.audit.length === 0 ? (
              <p className="muted">Nothing has happened yet.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">When</th>
                      <th scope="col">What happened</th>
                    </tr>
                  </thead>
                  <tbody>
                    {state.audit.map((a, i) => (
                      <tr key={i}>
                        <td style={{ whiteSpace: "nowrap" }}>{a.when}</td>
                        <td>
                          {a.event.charAt(0).toUpperCase() + a.event.slice(1)}
                          {a.detail && <div className="muted small">{a.detail}</div>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
