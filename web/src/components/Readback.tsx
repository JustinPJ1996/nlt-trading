// The readback: the engine's own plain-English statement of a strategy.
//
// This is the product's safety story -- a user who cannot read a spec can read
// "Buy NIFTY when RSI(14) crosses below 30" and say "that's wrong". So this
// component only *lays out* the text the server sent. Every line is shown, in
// order, with its words unchanged. The one cosmetic change is that the known
// section headings ("WHAT IT TRADES") are shown in sentence case.
//
// `labels`, when the server sends them, are one per body line in order: where
// that line came from. They are the server's words for it, never worked out here.

import type { ReadbackLabel } from "../api";

const LABEL_TEXT: Record<ReadbackLabel, string> = {
  said: "You said",
  worked_out: "We worked out",
  default: "Default",
};

const KNOWN_HEADING = /^(WHAT IT TRADES|WHEN IT [A-Z ]+|HOW MUCH|SAFETY LIMITS)$/;

type Block = { heading: string | null; lines: string[] };

export function splitReadback(text: string): { title: string; blocks: Block[] } {
  const rows = text.split("\n");
  const firstIdx = rows.findIndex((r) => r.trim() !== "");
  const title = firstIdx >= 0 ? rows[firstIdx].trim() : "";
  const blocks: Block[] = [];
  let current: Block | null = null;
  for (const row of rows.slice(firstIdx + 1)) {
    const trimmed = row.trim();
    if (!trimmed) {
      current = null;
      continue;
    }
    if (!row.startsWith(" ") && KNOWN_HEADING.test(trimmed)) {
      current = { heading: trimmed, lines: [] };
      blocks.push(current);
      continue;
    }
    if (!current) {
      current = { heading: null, lines: [] };
      blocks.push(current);
    }
    current.lines.push(trimmed);
  }
  return { title, blocks };
}

export function Readback({
  text,
  labels = null,
  compact = false,
  testId = "readback",
}: {
  text: string;
  labels?: (ReadbackLabel | null)[] | null;
  compact?: boolean;
  testId?: string;
}) {
  const { title, blocks } = splitReadback(text);
  const lineCount = blocks.reduce((n, b) => n + b.lines.length, 0);
  // Labels that do not line up with the text are dropped, not shifted.
  const usable = labels && labels.length === lineCount ? labels : null;
  let index = 0;
  return (
    <div className={`readback${compact ? " compact" : ""}`} data-testid={testId}>
      <p className="readback-title">{title}</p>
      {blocks.map((b, i) => (
        <div className="readback-section" key={i}>
          {b.heading && <p className="readback-heading">{b.heading}</p>}
          {b.lines.map((line, j) => {
            const label = usable ? usable[index] : null;
            index += 1;
            return (
              <p className="readback-line" key={j}>
                {line}
                {label && <span className={`readback-label ${label}`}>{LABEL_TEXT[label]}</span>}
              </p>
            );
          })}
        </div>
      ))}
      {usable && (
        <p className="readback-legend">
          <b>You said</b>: your own words. <b>We worked out</b>: filled in from what you said, such
          as the standard 14-period RSI. <b>Default</b>: you did not say, so we used our usual
          setting.
        </p>
      )}
    </div>
  );
}
