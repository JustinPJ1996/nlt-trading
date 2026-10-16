// Small helpers shared by several pages.

import { useEffect, useRef } from "react";
import { ApiError } from "./api";

// The server's own sentence when it sent one, otherwise our fallback.
export function message(err: unknown, fallback: string): string {
  return err instanceof ApiError ? err.message : fallback;
}

const TIME_FORMAT = new Intl.DateTimeFormat("en-GB", {
  day: "2-digit",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
  // Always Indian time, whatever the viewer's own clock says: every other
  // time on screen (and in the feed's own sentences) is IST.
  timeZone: "Asia/Kolkata",
});

export function when(iso: string): string {
  if (!iso) return "";
  try {
    return TIME_FORMAT.format(new Date(iso));
  } catch {
    return iso;
  }
}

// A run's health (as the server names it) to the colour class of its tag.
export function healthTagClass(health: string): string {
  switch (health) {
    case "success":
      return "good";
    case "warning":
      return "warning";
    case "error":
      return "critical";
    default:
      return "";
  }
}

// A status word for display. The server already sends friendly words for
// strategy state, run kind and run status; this only tidies anything that
// arrives raw ("not_connected" -> "Not connected") and leaves the rest alone.
const FRIENDLY: Record<string, string> = {
  draft: "Not tested",
  backtested: "Backtested",
  paper: "Paper trading",
  live: "Live",
  halted: "Halted",
  backtest: "Backtest",
  running: "Running",
  complete: "Finished",
  failed: "Failed",
  stopped: "Stopped",
};

export function friendly(value: string): string {
  if (!value) return value;
  if (value in FRIENDLY) return FRIENDLY[value];
  const spaced = value.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

// For a modal dialog: Escape closes it, and focus moves to the returned
// element (give it to the Close button) when it opens.
export function useDialog(onClose: () => void) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  useEffect(() => {
    closeRef.current?.focus();
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onCloseRef.current();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);
  return closeRef;
}
