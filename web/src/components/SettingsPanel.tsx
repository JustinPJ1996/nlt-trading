// The Settings panel: intraday or multi-day, when new entries stop, and when
// open positions are squared off. It only shows and edits numbers; every
// sentence about what they mean comes from the server's readback.

import type { SettingsEdits, StrategySettings } from "../api";

function Stepper({
  label,
  value,
  range,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  range: [number, number];
  disabled?: boolean;
  onChange: (n: number) => void;
}) {
  const [lo, hi] = range;
  const clamp = (n: number) => Math.min(hi, Math.max(lo, Math.round(n)));
  return (
    <span className="stepper">
      <button type="button" aria-label={`Less: ${label}`} disabled={disabled || value <= lo} onClick={() => onChange(clamp(value - 1))}>
        −
      </button>
      <input
        type="text"
        inputMode="numeric"
        aria-label={label}
        disabled={disabled}
        value={String(value)}
        onChange={(e) => {
          const n = parseInt(e.target.value.replace(/[^0-9]/g, ""), 10);
          if (Number.isFinite(n)) onChange(Math.min(hi, n));
        }}
        onBlur={() => onChange(clamp(value))}
      />
      <button type="button" aria-label={`More: ${label}`} disabled={disabled || value >= hi} onClick={() => onChange(clamp(value + 1))}>
        +
      </button>
    </span>
  );
}

export function SettingsPanel({
  settings,
  onChange,
  idPrefix = "settings",
}: {
  // The settings as they currently stand, with any unsaved changes applied.
  settings: StrategySettings;
  onChange: (edit: SettingsEdits) => void;
  idPrefix?: string;
}) {
  const intraday = settings.algo_type === "intraday";
  const { limits } = settings;
  return (
    <div className="settings-panel">
      <p className="label settings-heading">Settings</p>
      <label className="field mt-0">
        <span className="label">Trading style</span>
        <select
          value={settings.algo_type}
          disabled={!settings.can_be_intraday}
          onChange={(e) => onChange({ algo_type: e.target.value === "intraday" ? "intraday" : "multi_day" })}
        >
          <option value="intraday">Same-day (closes every day)</option>
          <option value="multi_day">Multi-day (can hold overnight)</option>
        </select>
        {!settings.can_be_intraday && <span className="hint">Daily candles are always multi-day.</span>}
      </label>

      <div className="field">
        <label className="check-row" htmlFor={`${idPrefix}-cutoff`}>
          <input
            id={`${idPrefix}-cutoff`}
            type="checkbox"
            checked={settings.entry_cutoff}
            onChange={(e) => onChange({ entry_cutoff: e.target.checked })}
          />
          <span className="label m-0">Entry cut-off time</span>
        </label>
        <span className="hint">No new entries within this many minutes before market close</span>
        <Stepper
          label="Entry cut-off, minutes before market close"
          value={settings.entry_cutoff_minutes}
          range={limits.entry_cutoff_minutes}
          disabled={!settings.entry_cutoff}
          onChange={(n) => onChange({ entry_cutoff_minutes: n })}
        />
        <span className="hint">
          Min: {limits.entry_cutoff_minutes[0]} mins, max: {limits.entry_cutoff_minutes[1]} mins before market close
        </span>
      </div>

      <div className="field">
        <label className="check-row" htmlFor={`${idPrefix}-eod`}>
          <input id={`${idPrefix}-eod`} type="checkbox" checked={intraday} disabled readOnly />
          <span className="label m-0">Square off all open positions at the end of the day</span>
        </label>
        <span className="hint">
          {intraday
            ? "Always on for same-day strategies. Switch the trading style to multi-day to turn it off."
            : "Off for multi-day strategies: positions are held overnight"}
        </span>
      </div>

      {intraday && (
        <div className="field">
          <span className="label">Minutes before market close</span>
          <span className="hint mt-0 mb-2">
            Time before market close at which open positions get squared off
          </span>
          <Stepper
            label="Square off, minutes before market close"
            value={settings.square_off_minutes}
            range={limits.square_off_minutes}
            onChange={(n) => onChange({ square_off_minutes: n })}
          />
          <span className="hint">
            Min: {limits.square_off_minutes[0]} mins, max: {limits.square_off_minutes[1]} mins before market close
          </span>
        </div>
      )}
      <p className="hint settings-note">{settings.close_note}</p>
    </div>
  );
}
