// The alerts inbox: every one-off alert that has fired, and every signal or
// exit from a strategy run as "alerts only". Nothing here moves money --
// this page only reads and marks notifications, both already built on the
// server (`/api/notifications`, `.../read`, `.../read-all`).

import { useCallback, useEffect, useState } from "react";
import { api, ApiError, type NotificationsState } from "../api";
import { ErrorLine, Working } from "../components/Callout";

function message(err: unknown, fallback: string) {
  return err instanceof ApiError ? err.message : fallback;
}

const TIME_FORMAT = new Intl.DateTimeFormat("en-GB", {
  day: "2-digit",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
  // Always Indian time, the same convention as every other time on screen.
  timeZone: "Asia/Kolkata",
});

function when(iso: string): string {
  if (!iso) return "";
  try {
    return TIME_FORMAT.format(new Date(iso));
  } catch {
    return iso;
  }
}

export function Alerts({ onChanged }: { onChanged?: () => void }) {
  const [state, setState] = useState<NotificationsState | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(() => {
    api.notifications().then(
      (s) => {
        setState(s);
        setError(null);
      },
      (err) => setError(message(err, "Could not load alerts.")),
    );
  }, []);

  useEffect(() => {
    reload();
    const timer = window.setInterval(reload, 30_000);
    return () => window.clearInterval(timer);
  }, [reload]);

  function markRead(id: number) {
    api.notificationRead(id).then(
      (s) => {
        setState(s);
        onChanged?.();
      },
      (err) => setError(message(err, "Could not mark that as read.")),
    );
  }

  function markAllRead() {
    api.notificationReadAll().then(
      (s) => {
        setState(s);
        onChanged?.();
      },
      (err) => setError(message(err, "Could not mark everything as read.")),
    );
  }

  return (
    <div className="narrow">
      <h1 className="page-title">Alerts</h1>
      <p className="page-lede">
        Every one-off alert that has fired, and every signal or exit from a strategy run as alerts only.
        Nothing here is a trade.
      </p>
      <ErrorLine error={error} />

      {!state && !error && <Working>Loading alerts...</Working>}

      {state && (
        <>
          {state.items.length > 0 && (
            <div className="btn-row" style={{ justifyContent: "flex-end", marginBottom: 16 }}>
              <button className="btn secondary small" onClick={markAllRead} disabled={state.unread_count === 0}>
                Mark all as read
              </button>
            </div>
          )}
          {state.items.length === 0 ? (
            <p className="muted">Nothing here yet.</p>
          ) : (
            <ul className="log" style={{ maxHeight: "none" }}>
              {state.items.map((n) => (
                <li
                  key={n.id}
                  className={n.read ? undefined : "clickable"}
                  onClick={n.read ? undefined : () => markRead(n.id)}
                >
                  <time>{when(n.when)}</time>
                  <span>
                    {!n.read && (
                      <span className="tag" style={{ marginRight: 8 }}>
                        unread
                      </span>
                    )}
                    <span style={{ fontWeight: n.read ? 400 : 650 }}>{n.text}</span>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
