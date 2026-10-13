// A user-set Slippage box: a kind toggle (percent of price, or rupees per
// unit) and an amount. Zero is the default everywhere a run can be started --
// a backtest, a paper run, or a one-off order -- and it is simply the honest
// absence of a number, not a hidden assumption the way 0.02% used to be.
//
// Changing this never touches the strategy itself: it is a run setting, like
// starting capital, so it must never hide the readback or withdraw "Yes,
// that's right" the way editing the sentence or the adjust boxes does.

import type { Slippage } from "../api";

export function SlippageInput({
  value,
  onChange,
  idPrefix = "slippage",
}: {
  value: Slippage;
  onChange: (next: Slippage) => void;
  idPrefix?: string;
}) {
  const problem =
    value.value < 0
      ? "Slippage cannot be negative."
      : value.kind === "pct" && value.value > 5
        ? "At most 5%."
        : value.kind === "rs" && value.value > 100
          ? "At most Rs 100."
          : null;
  return (
    <label className="field" style={{ marginTop: 0 }}>
      <span className="label">Slippage</span>
      <div className="choices" role="group" aria-label="Slippage kind">
        <button
          type="button"
          className={`choice${value.kind === "pct" ? " picked" : ""}`}
          aria-pressed={value.kind === "pct"}
          onClick={() => onChange({ ...value, kind: "pct" })}
        >
          % of price
        </button>
        <button
          type="button"
          className={`choice${value.kind === "rs" ? " picked" : ""}`}
          aria-pressed={value.kind === "rs"}
          onClick={() => onChange({ ...value, kind: "rs" })}
        >
          Rs per unit
        </button>
      </div>
      <span className="money" style={{ marginTop: 8 }}>
        <span aria-hidden="true">{value.kind === "pct" ? "%" : "Rs"}</span>
        <input
          type="text"
          inputMode="decimal"
          id={`${idPrefix}-amount`}
          value={value.value === 0 ? "" : String(value.value)}
          placeholder="0"
          onChange={(e) => {
            const raw = e.target.value.replace(/[^0-9.]/g, "");
            onChange({ ...value, value: raw === "" ? 0 : Number(raw) });
          }}
          aria-describedby={`${idPrefix}-hint`}
        />
      </span>
      <span className="hint" id={`${idPrefix}-hint`}>
        {problem ??
          (value.value === 0
            ? "Zero means every fill is tested at the exact price on the chart."
            : "Applied to every fill, in both directions.")}
      </span>
    </label>
  );
}
