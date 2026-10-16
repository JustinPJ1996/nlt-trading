// Results: the verdict first, in a sentence; then the one comparison that
// matters (your strategy against simply holding); then why to be careful;
// then the detail. Every word of the verdict, flags and notes comes from the
// engine's report -- this page only decides the order things appear in.

import { useState } from "react";
import { api, ApiError, type Result } from "../api";
import { Callout, ErrorLine, Working, type Tone } from "../components/Callout";
import { EquityChart } from "../components/EquityChart";
import { Readback } from "../components/Readback";

const FLAG_TONE: Record<string, Tone> = { critical: "critical", warning: "warning", info: "info" };
const TRADES_SHOWN = 12;

function capitalise(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function Results({
  result,
  onEdit,
  onSaved,
}: {
  result: Result | null;
  onEdit: () => void;
  onSaved: () => void;
}) {
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<Record<string, string>>({});
  const [saveError, setSaveError] = useState<string | null>(null);
  const [allTrades, setAllTrades] = useState(false);

  if (!result) {
    return (
      <div className="narrow">
        <h1 className="page-title">Results</h1>
        <div className="panel empty">
          <h2>Nothing tested yet</h2>
          <p className="muted mb-5">
            Describe a strategy, check what was understood, then run a backtest. The result appears here.
          </p>
          <button className="btn" onClick={onEdit}>
            Build a strategy
          </button>
        </div>
      </div>
    );
  }

  const { verdict, headline, params } = result;
  const intraday = result.tested.timeframe !== "1d";
  const benchmarkLabel = capitalise(headline.benchmark_name);
  const trades = allTrades ? result.trades : result.trades.slice(0, TRADES_SHOWN);
  const savedName = saved[result.result_id];

  async function save() {
    setSaving(true);
    setSaveError(null);
    try {
      const r = await api.save(result!.result_id);
      setSaved((s) => ({ ...s, [result!.result_id]: r.name }));
      onSaved();
    } catch (err) {
      setSaveError(err instanceof ApiError ? err.message : "Could not save.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div>
      {(result.window_notes ?? []).map((n, i) => (
        <p key={i} className="period">
          <strong>{n}</strong>
        </p>
      ))}
      <section className={`verdict${verdict.passed ? " passed" : ""}`} aria-labelledby="verdict">
        {/* `passed` means only that no *serious* warning sign fired -- a low bar
            a losing strategy can clear -- so it is never shown as a green "pass". */}
        <span className={`tag ${verdict.passed ? "" : "critical"}`}>
          <span aria-hidden="true">{verdict.passed ? "–" : "!"}</span>
          {verdict.passed ? "No serious warning signs" : "Serious warning signs"}
        </span>
        <h1 className="verdict-text" id="verdict">
          {verdict.summary}
        </h1>
      </section>

      <div className="versus">
        <div>
          <div className="figure-label">
            <span className="key s1" />
            Your strategy
          </div>
          <div className="figure-value">{headline.strategy_return}</div>
        </div>
        <div>
          <div className="figure-label">
            <span className="key s2" />
            {benchmarkLabel}
          </div>
          <div className="figure-value">{headline.benchmark_return}</div>
        </div>
      </div>
      <p className="period">
        {headline.first} to {headline.last}, starting with {params.capital_text}, charged{" "}
        {(() => {
          const charged = params.charges_applied ?? params.cost_model;
          return charged.charAt(0).toLowerCase() + charged.slice(1);
        })()}{" "}
        costs.
      </p>
      {params.slippage_text && <p className="period">{params.slippage_text}</p>}

      {verdict.flags.length > 0 && (
        <section className="section mt-6" aria-labelledby="flags">
          <h2 className="section-title" id="flags">
            Before you trust this
          </h2>
          {verdict.flags.map((f, i) => (
            <Callout key={i} tone={FLAG_TONE[f.severity] ?? "info"} title={f.headline}>
              <p>{f.detail}</p>
            </Callout>
          ))}
        </section>
      )}

      <section className="panel chart-panel" aria-label="Chart">
        <EquityChart points={result.chart} benchmarkLabel={benchmarkLabel} intraday={intraday} />
      </section>

      <div className="results-grid section">
        <section aria-labelledby="numbers">
          <h2 className="section-title" id="numbers">
            The numbers, side by side
          </h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">
                    <span className="visually-hidden">Measure</span>
                  </th>
                  <th scope="col" className="num">
                    Your strategy
                  </th>
                  <th scope="col" className="num">
                    {benchmarkLabel}
                  </th>
                </tr>
              </thead>
              <tbody>
                {result.metrics.map((m) => (
                  <tr key={m.label}>
                    <th scope="row">{m.label}</th>
                    <td className="num">{m.strategy}</td>
                    <td className="num">{m.benchmark || <span className="muted">Not applicable</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {result.before_after_charges && (
            <details className="fold mt-6">
              <summary>
                <span className="section-title" id="charges">
                  Before and after charges
                </span>
              </summary>
              <p className="hint mb-3">
                The same strategy tested twice: once with no charges at all, once with brokerage,
                taxes and fees deducted. The gap is what trading costs took.
              </p>
              <div className="table-wrap">
                <table aria-labelledby="charges">
                  <thead>
                    <tr>
                      <th scope="col">
                        <span className="visually-hidden">Measure</span>
                      </th>
                      <th scope="col" className="num">
                        Before charges
                      </th>
                      <th scope="col" className="num">
                        After charges
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.before_after_charges.map((row) => (
                      <tr key={row.label}>
                        <th scope="row">{row.label}</th>
                        <td className="num">{row.before}</td>
                        <td className="num">{row.after}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </section>

        <section aria-labelledby="tested">
          <h2 className="section-title" id="tested">
            What was tested
          </h2>
          <Readback text={result.tested.readback} compact />
        </section>
      </div>

      {result.notes.length > 0 && (
        <section className="section" aria-labelledby="notes">
          <h2 className="section-title" id="notes">
            Things you should know about this test
          </h2>
          {result.notes.map((n, i) => (
            <Callout key={i} tone="warning">
              {n}
            </Callout>
          ))}
        </section>
      )}

      {(result.assumptions.length > 0 || result.weak_spots.length > 0) && (
        <details className="section method fold">
          <summary>
            <span className="section-title" id="method">
              How this was tested
            </span>
            <span className="fold-hint">What the test assumed, and where it could be wrong</span>
          </summary>
          <div className="method-grid">
            <div>
              <h3 className="method-title">What it assumed</h3>
              <ul className="method-list">
                {result.assumptions.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="method-title">Weak spots of this test</h3>
              <ul className="method-list">
                {result.weak_spots.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          </div>
        </details>
      )}

      <section className="section" aria-labelledby="trades">
        <h2 className="section-title" id="trades">
          Every trade{result.trades.length ? ` (${result.trades.length})` : ""}
        </h2>
        {result.trades.length > 0 && (
          <p className="mb-3">
            <a
              className="btn secondary no-underline"
              href={`/api/results/${encodeURIComponent(result.result_id)}/trades.csv`}
              download
            >
              Download trades (CSV)
            </a>
          </p>
        )}
        {result.trades.length === 0 ? (
          <p className="muted">This strategy never took a trade in the period tested.</p>
        ) : (
          <>
            <div className="table-wrap trades-wrap">
              <table className="trades-table">
                <thead>
                  <tr>
                    <th scope="col">Entered</th>
                    <th scope="col">Exited</th>
                    <th scope="col" className="num">
                      Bought / sold at
                    </th>
                    <th scope="col">Why it closed</th>
                    <th scope="col" className="num">
                      Result after costs
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map((t, i) => (
                    <tr key={i}>
                      <td className="trade-entered">
                        {t.entered}
                        <div className="muted small">
                          {t.direction} {t.symbol}
                        </div>
                        {t.why && <div className="muted small">{t.why}</div>}
                        {t.legs && t.legs.length > 0 && (
                          <ul className="muted small leg-list">
                            {t.legs.map((l, j) => (
                              <li key={j}>
                                {l.side} {l.quantity} {l.label}
                                {l.reentry > 0 ? ` (re-entry ${l.reentry})` : ""} at {l.entry_price}, closed{" "}
                                {l.exited} at {l.exit_price} – {l.reason}
                                {l.stop ? ` (stop ${l.stop})` : ""}:{" "}
                                <span className={l.pnl > 0 ? "pos" : l.pnl < 0 ? "neg" : ""}>{l.pnl_text}</span>
                                <div>{l.why}</div>
                              </li>
                            ))}
                          </ul>
                        )}
                      </td>
                      <td data-label="Exited">{t.exited}</td>
                      <td className="num" data-label="Bought / sold at">
                        {t.entry_price}
                        <div className="muted small">{t.exit_price}</div>
                      </td>
                      <td data-label="Why it closed">{t.reason}</td>
                      <td className={`num trade-result ${t.pnl > 0 ? "pos" : t.pnl < 0 ? "neg" : ""}`} data-label="Result after costs">
                        {t.pnl_text}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {result.trades.length > TRADES_SHOWN && (
              <button className="btn quiet mt-2" onClick={() => setAllTrades((v) => !v)}>
                {allTrades ? "Show fewer trades" : `Show all ${result.trades.length} trades`}
              </button>
            )}
          </>
        )}
      </section>

      <div className="save-bar">
        <div>
          {savedName ? (
            <p>
              <strong>Saved.</strong> Find "{savedName}" in <a href="#/strategies">My strategies</a>, where
              you can also start paper trading it.
            </p>
          ) : (
            <p className="muted save-note">Save this strategy to paper trade it on live prices later.</p>
          )}
          <ErrorLine error={saveError} />
        </div>
        <div className="btn-row">
          <button className="btn secondary" onClick={onEdit}>
            Change the strategy
          </button>
          {!savedName && (
            <button className="btn" onClick={save} disabled={saving}>
              Save this strategy
            </button>
          )}
          {saving && <Working>Saving...</Working>}
        </div>
      </div>
    </div>
  );
}
