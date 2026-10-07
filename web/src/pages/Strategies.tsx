import { useEffect, useState } from "react";
import { api, ApiError, type SavedStrategy, type Understood } from "../api";
import { ErrorLine, Working } from "../components/Callout";
import { Readback } from "../components/Readback";

export function Strategies({
  onTestAgain,
  onBuild,
}: {
  onTestAgain: (description: string, understood: Understood) => void;
  onBuild: () => void;
}) {
  const [list, setList] = useState<SavedStrategy[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [opening, setOpening] = useState<number | null>(null);

  useEffect(() => {
    api.strategies().then(
      (r) => setList(r.strategies),
      (err) => setError(err instanceof ApiError ? err.message : "Could not load your strategies."),
    );
  }, []);

  async function testAgain(id: number) {
    setOpening(id);
    setError(null);
    try {
      const r = await api.savedSpec(id);
      onTestAgain(r.description, r.understood);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not open that strategy.");
      setOpening(null);
    }
  }

  return (
    <div>
      <h1 className="page-title">My strategies</h1>
      <p className="page-lede">
        A strategy is <strong>proven</strong> once it has a finished backtest and a finished paper-trading
        run. Only proven strategies will ever be allowed to trade for real.
      </p>
      <ErrorLine error={error} />
      {list === null && !error && <Working>Loading...</Working>}
      {list && list.length === 0 && (
        <div className="panel empty">
          <h2>No saved strategies yet</h2>
          <p className="muted" style={{ marginBottom: 20 }}>
            Build a strategy, test it, then press "Save this strategy" on its results.
          </p>
          <button className="btn" onClick={onBuild}>
            Build a strategy
          </button>
        </div>
      )}
      {list && list.length > 0 && (
        <div className="list">
          {list.map((s) => (
            <details className="list-item" key={s.id}>
              <summary>
                <span className="item-head">
                  <span className="item-name">{s.name}</span>
                  <span className="item-meta">Version {s.version}</span>
                  <span className="tag">{s.state}</span>
                  {s.proven ? (
                    <span className="tag good">
                      <span aria-hidden="true">{"✓"}</span> Proven
                    </span>
                  ) : (
                    <span className="tag">Not proven yet</span>
                  )}
                  <span className="item-meta" style={{ marginLeft: "auto" }}>
                    Saved {s.created}
                  </span>
                </span>
              </summary>
              <div className="list-item-body stack">
                <p className="muted">You wrote: "{s.description}"</p>
                <Readback text={s.readback} compact />
                {s.runs.length > 0 && (
                  <div className="table-wrap" style={{ marginTop: 16 }}>
                    <table>
                      <thead>
                        <tr>
                          <th scope="col">Run</th>
                          <th scope="col">Started</th>
                          <th scope="col">Status</th>
                          <th scope="col" className="num">
                            Money
                          </th>
                          <th scope="col" className="num">
                            Return
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {s.runs.map((r) => (
                          <tr key={r.id}>
                            <td>{r.kind}</td>
                            <td>{r.started}</td>
                            <td>
                              {r.status}
                              {r.error && <div className="small neg">{r.error}</div>}
                            </td>
                            <td className="num">{r.capital_text}</td>
                            <td className="num">{r.return_text}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                <div className="btn-row" style={{ marginTop: 16 }}>
                  <button className="btn secondary" onClick={() => testAgain(s.id)} disabled={opening !== null}>
                    Test again
                  </button>
                  <a className="btn secondary" href="#/paper" style={{ textDecoration: "none" }}>
                    Paper trade it
                  </a>
                  {opening === s.id && <Working>Opening...</Working>}
                </div>
              </div>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}
