// Home: the first thing a new trader sees. One question, one box, a sentence
// taken apart to show what to write, and ready-made strategies with their real
// numbers. Whatever is written or picked here goes to Build, where it is read
// back and confirmed before anything is tested.

import { useState, type FormEvent } from "react";
import { Anatomy } from "../components/Anatomy";
import { Library } from "../components/Library";

export function Home({ onStart }: { onStart: (text: string) => void }) {
  const [text, setText] = useState("");

  function submit(e: FormEvent) {
    e.preventDefault();
    if (text.trim()) onStart(text.trim());
  }

  return (
    <div>
      <h1 className="home-title">What do you want to test?</h1>
      <p className="page-lede">
        Describe a trading idea in plain English. You will see exactly how it was read before
        anything is tested on past prices, with pretend money.
      </p>
      <form className="home-ask narrow" onSubmit={submit}>
        <label className="visually-hidden" htmlFor="home-describe">
          Your strategy, in plain English
        </label>
        <textarea
          id="home-describe"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(e);
          }}
          placeholder="e.g. buy nifty when rsi cracks 30, target 2%, stop loss 1%"
          rows={3}
        />
        <div className="btn-row mt-4">
          <button className="btn" type="submit" disabled={!text.trim()}>
            Check my strategy
          </button>
        </div>
        <Anatomy />
      </form>
      <Library onPick={onStart} />
    </div>
  );
}
