// Number and date wording for the few values the browser formats itself --
// the chart's axes and hover readout. Everything else arrives already worded
// by the Python side, so rupees read the same on every screen.

const IST = "Asia/Kolkata";

export function formatDate(iso: string, withTime: boolean): string {
  const d = new Date(iso);
  const date = d.toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    timeZone: IST,
  });
  if (!withTime) return date;
  const time = d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: IST });
  return `${date}, ${time}`;
}

export function monthLabel(iso: string): string {
  return new Date(iso).toLocaleDateString("en-GB", { month: "short", year: "2-digit", timeZone: IST });
}

/** Rs 1,00,000 -- Indian digit grouping, no paise. */
export function rupeesFull(v: number): string {
  const sign = v < 0 ? "-" : "";
  return `${sign}Rs ${Math.round(Math.abs(v)).toLocaleString("en-IN")}`;
}

/** Axis labels: Rs 1.2 Cr, Rs 85 L, Rs 40K. */
export function rupeesCompact(v: number): string {
  const a = Math.abs(v);
  const sign = v < 0 ? "-" : "";
  const trim = (x: number) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1).replace(/\.0$/, "") : x.toFixed(2).replace(/\.?0+$/, ""));
  if (a >= 1e7) return `${sign}Rs ${trim(a / 1e7)} Cr`;
  if (a >= 1e5) return `${sign}Rs ${trim(a / 1e5)} L`;
  if (a >= 1e3) return `${sign}Rs ${trim(a / 1e3)}K`;
  return `${sign}Rs ${Math.round(a)}`;
}

/** What the user typed into a money box, as a number; "" or junk is null. */
export function parseRupees(text: string): number | null {
  const cleaned = text.replace(/[^0-9.]/g, "");
  if (!cleaned) return null;
  const v = Number(cleaned);
  return Number.isFinite(v) ? v : null;
}
