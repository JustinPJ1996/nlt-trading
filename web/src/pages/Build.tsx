// Build: describe it, check what was understood, test it.
//
// Two rules this page keeps, both about the readback being trustworthy:
//  - If the sentence changes after it was checked, the readback and the test
//    step disappear until it is checked again. A readback is only ever shown
//    beside the sentence it was read from.
//  - "Run backtest" sends the confirmed spec itself, never the sentence, so
//    what gets tested is exactly what the user said yes to.
// A suggested answer to a question is shown as a hint, never pre-filled: an
// answer the user did not type is an answer they did not give -- the same
// reason the engine never invents a stop loss.

import { useEffect, useRef, useState, type Dispatch, type FormEvent, type SetStateAction } from "react";
import { api, ApiError, type Options, type Result, type Translation, type Understood } from "../api";
import { Callout, ErrorLine, Working } from "../components/Callout";
import { Readback } from "../components/Readback";
import { parseRupees } from "../format";

export type BuildState = {
  text: string;
  checkedText: string | null;
  translation: Translation | null;
  answers: Record<string, string>;
  draftAnswers: Record<string, string>;
  confirmed: boolean;
  fromSaved: boolean;
  capital: string;
  costModel: string;
  start: string;
  end: string;
};

export function emptyBuild(): BuildState {
  return {
    text: "",
    checkedText: null,
    translation: null,
    answers: {},
    draftAnswers: {},
    confirmed: false,
    fromSaved: false,
    capital: "",
    costModel: "",
    start: "",
    end: "",
  };
}

export function Build({
  options,
  state,
  setState,
  onResult,
}: {
  options: Options | null;
  state: BuildState;
  setState: Dispatch<SetStateAction<BuildState>>;
  onResult: (r: Result) => void;
}) {
  const [checking, setChecking] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const step2Ref = useRef<HTMLLIElement>(null);
  const step3Ref = useRef<HTMLLIElement>(null);

  const set = (patch: Partial<BuildState>) => setState((s) => ({ ...s, ...patch }));

  // Fill the test settings once the server has said what the defaults are.
  useEffect(() => {
    if (options && !state.costModel) {
      set({
        capital: String(options.default_capital),
        costModel: options.cost_models[0],
        start: options.default_start,
        end: options.default_end,
      });
    }
  }, [options]); // eslint-disable-line react-hooks/exhaustive-deps

  const stale = state.checkedText !== null && state.text.trim() !== state.checkedText;
  const tr = stale ? null : state.translation;
  const understood: Understood | null = tr?.understood ?? null;
  const confirmed = state.confirmed && !!understood;

  async function check(text: string, answers: Record<string, string>) {
    const clean = text.trim();
    if (!clean) {
      setError("Describe a strategy first.");
      return;
    }
    setChecking(true);
    setError(null);
    setRunError(null);
    try {
      const t = await api.translate(clean, answers);
      setState((s) => ({
        ...s,
        checkedText: clean,
        translation: t,
        answers: t.answers,
        draftAnswers: {},
        confirmed: false,
        fromSaved: false,
      }));
      requestAnimationFrame(() => step2Ref.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not check that.");
    } finally {
      setChecking(false);
    }
  }

  function onDescribe(e: FormEvent) {
    e.preventDefault();
    // A changed sentence starts its questions afresh; answers belong to the words they answered.
    const sameText = state.text.trim() === state.checkedText;
    check(state.text, sameText ? state.answers : {});
  }

  function onAnswers(e: FormEvent) {
    e.preventDefault();
    check(state.text, { ...state.answers, ...state.draftAnswers });
  }

  function confirm() {
    set({ confirmed: true });
    requestAnimationFrame(() => step3Ref.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
  }

  function rephrase() {
    set({ confirmed: false });
    textRef.current?.focus();
    textRef.current?.select();
  }

  const capital = parseRupees(state.capital);
  const capitalProblem =
    capital === null ? "Enter an amount." : capital < 1000 ? "Use at least Rs 1,000." : null;
  const dateProblem = state.start && state.end && state.start >= state.end ? "The start date must be before the end date." : null;

  async function run(e: FormEvent) {
    e.preventDefault();
    if (!understood || capitalProblem || dateProblem) return;
    setRunning(true);
    setRunError(null);
    try {
      const r = await api.backtest({
        spec: understood.spec,
        capital: capital!,
        cost_model: state.costModel,
        start: state.start,
        end: state.end,
      });
      onResult(r);
    } catch (err) {
      setRunError(err instanceof ApiError ? err.message : "The backtest did not finish.");
    } finally {
      setRunning(false);
    }
  }

  const step2Class = tr ? (confirmed ? "done" : "active") : "waiting";
  const step3Class = confirmed ? "active" : "waiting";
  const visibleQuestions = tr?.questions ?? [];

  return (
    <div className="narrow">
      <h1 className="page-title">Build a strategy</h1>
      <p className="page-lede">
        Write it the way you would explain it to a friend. You will see exactly what was understood
        before anything is tested.
      </p>

      <ol className="steps">
        <li className={`step ${tr ? "done" : "active"}`}>
          <h2 className="step-title">Describe your strategy</h2>
          <form className="describe" onSubmit={onDescribe}>
            <label className="visually-hidden" htmlFor="describe">
              Your strategy, in plain English
            </label>
            <textarea
              id="describe"
              ref={textRef}
              value={state.text}
              // Any edit withdraws the approval: it was given to the words as they were.
              onChange={(e) => set({ text: e.target.value, confirmed: false })}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) onDescribe(e);
              }}
              placeholder="e.g. buy nifty when rsi cracks 30, target 2%, stop loss 1%"
              rows={3}
            />
            {options && !state.text && (
              <div className="examples" aria-label="Examples">
                <span className="muted small" style={{ alignSelf: "center" }}>
                  Or try:
                </span>
                {options.examples.map((ex) => (
                  <button type="button" className="example" key={ex} onClick={() => {
                      set({ text: ex });
                      check(ex, {});
                    }}>
                    {ex}
                  </button>
                ))}
              </div>
            )}
            <div className="btn-row" style={{ marginTop: 16 }}>
              <button className="btn" type="submit" disabled={checking || !state.text.trim()}>
                Check my strategy
              </button>
              {checking && <Working>Reading your strategy...</Working>}
            </div>
            <div style={{ marginTop: 12 }}>
              <ErrorLine error={error} />
            </div>
          </form>
        </li>

        <li className={`step ${step2Class}`} ref={step2Ref}>
          <h2 className="step-title">Check what I understood</h2>
          {stale && (
            <p className="muted">You changed your description. Check it again to see what I understand now.</p>
          )}
          {tr && (
            <div className="stack reveal">
              {tr.unparsed.length > 0 && (
                <Callout tone="critical" title="I did not understand part of this">
                  {tr.unparsed.map((u, i) => (
                    <p key={i}>{u}</p>
                  ))}
                </Callout>
              )}
              {tr.notes.map((n, i) => (
                <Callout key={i} tone="info">
                  {n}
                </Callout>
              ))}

              {understood && (
                <>
                  {state.fromSaved ? (
                    <p className="muted">This is the saved strategy. Read it again before testing.</p>
                  ) : (
                    <p className="muted">Read this carefully. If anything is not what you meant, change your description.</p>
                  )}
                  <Readback text={understood.readback} />
                  {tr.read_by_ai && (
                    <p className="ai-note">
                      An AI model helped read your sentence. The description above is written by the
                      app from what it will actually test, not by the AI.
                    </p>
                  )}
                  {!confirmed && (
                    <div className="btn-row" style={{ marginTop: 20 }}>
                      <button className="btn" onClick={confirm}>
                        Yes, that's right
                      </button>
                      <button className="btn secondary" onClick={rephrase}>
                        No, change my description
                      </button>
                    </div>
                  )}
                </>
              )}

              {!understood && visibleQuestions.length > 0 && (
                <form className="panel" onSubmit={onAnswers}>
                  <p style={{ fontWeight: 650, marginBottom: 16 }}>I need a little more detail first.</p>
                  {visibleQuestions.map((q) => (
                    <div className="question" key={q.field}>
                      <label className="field">
                        <span className="label">{q.text}</span>
                        <input
                          type="text"
                          value={state.draftAnswers[q.field] ?? ""}
                          onChange={(e) => set({ draftAnswers: { ...state.draftAnswers, [q.field]: e.target.value } })}
                        />
                        <span className="hint">
                          {q.why}
                          {q.suggestion ? ` ${q.suggestion}` : ""}
                        </span>
                      </label>
                    </div>
                  ))}
                  <div className="btn-row" style={{ marginTop: 8 }}>
                    <button
                      className="btn"
                      type="submit"
                      disabled={checking || !visibleQuestions.some((q) => (state.draftAnswers[q.field] ?? "").trim())}
                    >
                      Use these answers
                    </button>
                    {checking && <Working>Reading again...</Working>}
                  </div>
                </form>
              )}

              {!understood && visibleQuestions.length === 0 && tr.unparsed.length === 0 && (
                <Callout tone="warning" title="I could not build a strategy from that">
                  Try describing when to buy or sell, and where the stop loss and target are.
                </Callout>
              )}
            </div>
          )}
        </li>

        <li className={`step ${step3Class}`} ref={step3Ref}>
          <h2 className="step-title">Test it on past prices</h2>
          {confirmed && options && (
            <form className="panel reveal" onSubmit={run}>
              <div className="grid-2">
                <label className="field" style={{ marginTop: 0 }}>
                  <span className="label">Starting money</span>
                  <span className="money">
                    <span aria-hidden="true">Rs</span>
                    <input
                      type="text"
                      inputMode="numeric"
                      value={state.capital ? Number(parseRupees(state.capital) ?? 0).toLocaleString("en-IN") : ""}
                      onChange={(e) => set({ capital: e.target.value.replace(/[^0-9]/g, "") })}
                      aria-describedby="capital-hint"
                    />
                  </span>
                  <span className="hint" id="capital-hint">
                    {capitalProblem ?? "Pretend money. Nothing real is traded."}
                  </span>
                </label>
                <label className="field" style={{ marginTop: 0 }}>
                  <span className="label">Trading costs to charge</span>
                  <select value={state.costModel} onChange={(e) => set({ costModel: e.target.value })}>
                    {options.cost_models.map((c) => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                  <span className="hint">Shares and futures always use their own real charges.</span>
                </label>
                <label className="field" style={{ marginTop: 0 }}>
                  <span className="label">From</span>
                  <input type="date" value={state.start} onChange={(e) => set({ start: e.target.value })} />
                </label>
                <label className="field" style={{ marginTop: 0 }}>
                  <span className="label">To</span>
                  <input type="date" value={state.end} onChange={(e) => set({ end: e.target.value })} />
                </label>
              </div>
              {dateProblem && (
                <div style={{ marginTop: 12 }}>
                  <ErrorLine error={dateProblem} />
                </div>
              )}
              <p className="hint" style={{ marginTop: 12 }}>
                A longer stretch of history gives a more trustworthy answer. Past results are not a
                promise.
              </p>
              <div className="btn-row" style={{ marginTop: 20 }}>
                <button className="btn" type="submit" disabled={running || !!capitalProblem || !!dateProblem}>
                  Run backtest
                </button>
                {running && <Working>Testing on past prices. This can take a minute...</Working>}
              </div>
              {runError && (
                <div style={{ marginTop: 16 }}>
                  <ErrorLine error={runError} />
                </div>
              )}
            </form>
          )}
        </li>
      </ol>
    </div>
  );
}
