// The readback: the engine's own plain-English statement of a strategy.
//
// This is the product's safety story -- a user who cannot read a spec can read
// "Buy NIFTY when RSI(14) crosses below 30" and say "that's wrong". So this
// component only *lays out* the text the server sent. Every line is shown, in
// order, with its words unchanged. The one cosmetic change is that the known
// section headings ("WHAT IT TRADES") are shown in sentence case.

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

export function Readback({ text, compact = false }: { text: string; compact?: boolean }) {
  const { title, blocks } = splitReadback(text);
  return (
    <div className={`readback${compact ? " compact" : ""}`} data-testid="readback">
      <p className="readback-title">{title}</p>
      {blocks.map((b, i) => (
        <div className="readback-section" key={i}>
          {b.heading && <p className="readback-heading">{b.heading}</p>}
          {b.lines.map((line, j) => (
            <p className="readback-line" key={j}>
              {line}
            </p>
          ))}
        </div>
      ))}
    </div>
  );
}
