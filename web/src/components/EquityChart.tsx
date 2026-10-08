// Strategy vs. holding, on one rupee axis, with the drop-from-peak beneath it.
//
// The two panels share their x positions, so a dip in the lower panel always
// sits under the fall in the upper one that caused it. Both panels use the same
// two colours for the same two things (dataviz slots 1 and 2), there is always
// a legend plus direct labels at the line ends, and hovering -- or focusing the
// chart and using the arrow keys -- reads out every series at that date.

import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { ChartPoint } from "../api";
import { formatDate, monthLabel, rupeesCompact, rupeesFull } from "../format";

const TOP_H = 260;
const GAP = 44;
const DD_H = 110;
const PAD_L = 64;
const PAD_R = 12;
const PAD_T = 24;
const PAD_B = 28;

function niceTicks(min: number, max: number, count = 4): number[] {
  if (!(max > min)) return [min];
  const raw = (max - min) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(v);
  return out;
}

function timeTicks(points: ChartPoint[]): { idx: number[]; byYear: boolean } {
  if (points.length < 2) return { idx: [], byYear: true };
  const starts = (key: (t: string) => string) => {
    const keys = points.map((p) => key(p.t));
    return keys.flatMap((k, i) => (i > 0 && k !== keys[i - 1] ? [i] : []));
  };
  const years = starts((t) => t.slice(0, 4));
  if (years.length >= 2) {
    const every = Math.ceil(years.length / 7);
    return { idx: years.filter((_, i) => i % every === 0), byYear: true };
  }
  const months = starts((t) => t.slice(0, 7));
  const every = Math.max(1, Math.ceil(months.length / 6));
  return { idx: months.filter((_, i) => i % every === 0), byYear: false };
}

function path(values: number[], x: (i: number) => number, y: (v: number) => number): string {
  return values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
}

export function EquityChart({
  points,
  benchmarkLabel,
  intraday,
}: {
  points: ChartPoint[];
  benchmarkLabel: string;
  intraday: boolean;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(280, entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const n = points.length;
  const plotW = width - PAD_L - PAD_R;
  const ddTop = PAD_T + TOP_H + GAP;
  const height = ddTop + DD_H + PAD_B;

  const geo = useMemo(() => {
    const eq = points.flatMap((p) => [p.s, p.b]);
    let lo = Math.min(...eq);
    let hi = Math.max(...eq);
    const pad = (hi - lo) * 0.06 || hi * 0.02 || 1;
    lo -= pad;
    hi += pad;
    const ddMin = Math.min(-1, ...points.flatMap((p) => [p.sd, p.bd]));
    const x = (i: number) => PAD_L + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW);
    const yEq = (v: number) => PAD_T + TOP_H - ((v - lo) / (hi - lo)) * TOP_H;
    const yDd = (v: number) => ddTop + (v / ddMin) * DD_H;
    return {
      x,
      yEq,
      yDd,
      eqTicks: niceTicks(lo, hi),
      ddTicks: niceTicks(ddMin, 0, 2),
      xTicks: timeTicks(points),
    };
  }, [points, n, plotW, ddTop]);

  if (n === 0) return null;
  const { x, yEq, yDd } = geo;

  const last = points[n - 1];
  // Direct labels at the line ends, nudged apart when the lines finish close together.
  let ys = yEq(last.s) - 8;
  let yb = yEq(last.b) - 8;
  if (Math.abs(ys - yb) < 16) {
    if (ys <= yb) ys = yb - 16;
    else yb = ys - 16;
  }

  function pick(clientX: number) {
    const rect = wrapRef.current!.getBoundingClientRect();
    const rel = (clientX - rect.left - PAD_L) / plotW;
    setHover(Math.max(0, Math.min(n - 1, Math.round(rel * (n - 1)))));
  }

  function onKey(e: KeyboardEvent) {
    const step = e.shiftKey ? Math.max(1, Math.round(n / 20)) : 1;
    if (e.key === "ArrowRight") setHover((h) => Math.min(n - 1, (h ?? -1) + step));
    else if (e.key === "ArrowLeft") setHover((h) => Math.max(0, (h ?? n) - step));
    else if (e.key === "Home") setHover(0);
    else if (e.key === "End") setHover(n - 1);
    else if (e.key === "Escape") setHover(null);
    else return;
    e.preventDefault();
  }

  const h = hover !== null ? points[hover] : null;
  const tipLeft = hover !== null ? x(hover) : 0;
  const tipFlip = tipLeft > width - 240;

  return (
    <div>
      <div className="chart-legend">
        <span>
          <span className="key" style={{ background: "var(--series-1)" }} />
          Your strategy
        </span>
        <span>
          <span className="key" style={{ background: "var(--series-2)" }} />
          {benchmarkLabel}
        </span>
      </div>
      <div
        className="chart-wrap"
        ref={wrapRef}
        onPointerMove={(e: PointerEvent) => pick(e.clientX)}
        onPointerDown={(e: PointerEvent) => pick(e.clientX)}
        onPointerLeave={() => setHover(null)}
      >
        <svg
          viewBox={`0 0 ${width} ${height}`}
          height={height}
          role="img"
          tabIndex={0}
          aria-label={`Account value over time for your strategy and for ${benchmarkLabel}, with the drop from each one's previous peak underneath. Use the left and right arrow keys to read values.`}
          onKeyDown={onKey}
          onBlur={() => setHover(null)}
        >
          <text className="chart-panel-title" x={PAD_L} y={PAD_T - 10}>
            Account value
          </text>
          <g className="chart-grid">
            {geo.eqTicks.map((t) => (
              <line key={t} x1={PAD_L} x2={width - PAD_R} y1={yEq(t)} y2={yEq(t)} />
            ))}
          </g>
          <g className="chart-axis">
            {geo.eqTicks.map((t) => (
              <text key={t} x={PAD_L - 10} y={yEq(t) + 4} textAnchor="end">
                {rupeesCompact(t)}
              </text>
            ))}
          </g>

          <path d={path(points.map((p) => p.b), x, yEq)} fill="none" stroke="var(--series-2)" strokeWidth={2} strokeLinejoin="round" />
          <path d={path(points.map((p) => p.s), x, yEq)} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinejoin="round" />

          <g className="chart-axis chart-label" style={{ fontWeight: 600 }}>
            <text x={width - PAD_R} y={ys} textAnchor="end" style={{ fill: "var(--ink)" }}>
              Your strategy
            </text>
            <text x={width - PAD_R} y={yb} textAnchor="end" style={{ fill: "var(--ink-2)" }}>
              {benchmarkLabel}
            </text>
          </g>

          <text className="chart-panel-title" x={PAD_L} y={ddTop - 10}>
            Drop from the previous high
          </text>
          <g className="chart-grid">
            {geo.ddTicks.map((t) => (
              <line key={t} x1={PAD_L} x2={width - PAD_R} y1={yDd(t)} y2={yDd(t)} className={t === 0 ? "chart-zero" : undefined} />
            ))}
          </g>
          <g className="chart-axis">
            {geo.ddTicks.map((t) => (
              <text key={t} x={PAD_L - 10} y={yDd(t) + 4} textAnchor="end">
                {t === 0 ? "0%" : `${Math.round(t)}%`}
              </text>
            ))}
          </g>
          <path d={path(points.map((p) => p.bd), x, yDd)} fill="none" stroke="var(--series-2)" strokeWidth={2} strokeLinejoin="round" />
          <path d={path(points.map((p) => p.sd), x, yDd)} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinejoin="round" />

          <g className="chart-axis">
            {geo.xTicks.idx.map((i) => (
              <text key={i} x={x(i)} y={height - 6} textAnchor="middle">
                {geo.xTicks.byYear ? points[i].t.slice(0, 4) : monthLabel(points[i].t)}
              </text>
            ))}
          </g>

          {h && hover !== null && (
            <g pointerEvents="none">
              <line x1={x(hover)} x2={x(hover)} y1={PAD_T} y2={ddTop + DD_H} stroke="var(--ink-3)" strokeWidth={1} />
              {[
                [yEq(h.b), "var(--series-2)"],
                [yEq(h.s), "var(--series-1)"],
                [yDd(h.bd), "var(--series-2)"],
                [yDd(h.sd), "var(--series-1)"],
              ].map(([cy, color], k) => (
                <circle key={k} cx={x(hover)} cy={cy as number} r={4.5} fill={color as string} stroke="var(--surface)" strokeWidth={2} />
              ))}
            </g>
          )}
        </svg>

        {h && (
          <div
            className="chart-tip"
            style={tipFlip ? { right: width - tipLeft + 14 } : { left: tipLeft + 14 }}
            aria-live="polite"
          >
            <div className="chart-tip-date">{formatDate(h.t, intraday)}</div>
            <div className="chart-tip-row">
              <span className="key" style={{ background: "var(--series-1)" }} />
              <span>Your strategy</span>
              <strong>{rupeesFull(h.s)}</strong>
              <span className="key" style={{ background: "var(--series-2)" }} />
              <span>{benchmarkLabel}</span>
              <strong>{rupeesFull(h.b)}</strong>
            </div>
            <div className="chart-tip-sub">Drop from the previous high</div>
            <div className="chart-tip-row">
              <span className="key" style={{ background: "var(--series-1)" }} />
              <span>Your strategy</span>
              <strong>{h.sd.toFixed(1)}%</strong>
              <span className="key" style={{ background: "var(--series-2)" }} />
              <span>{benchmarkLabel}</span>
              <strong>{h.bd.toFixed(1)}%</strong>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
