// One example sentence taken apart, so a new trader sees what a complete
// strategy says. Stop loss is marked required: a strategy without one comes
// back as a question, never with a stop made up for it.

const PARTS: { words: string; name: string; required?: boolean }[] = [
  { words: "Buy NIFTY", name: "What to trade" },
  { words: "when RSI(14) crosses below 30,", name: "When to buy" },
  { words: "target 2%,", name: "When to sell" },
  { words: "stop loss 1%,", name: "Stop loss", required: true },
  { words: "on 15-minute candles,", name: "Chart" },
  { words: "backtest the last 2 years", name: "Test period" },
];

export function Anatomy() {
  return (
    <div className="anatomy" aria-label="What a complete strategy says, with an example">
      {PARTS.map((p) => (
        <span className="anatomy-part" key={p.name}>
          <span className="anatomy-words">{p.words}</span>
          <span className="anatomy-name">
            {p.name}
            {p.required && <b> · required</b>}
          </span>
        </span>
      ))}
    </div>
  );
}
