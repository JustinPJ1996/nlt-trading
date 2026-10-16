// The alerts inbox: every one-off alert that has fired, and every signal or
// exit from a strategy run as "alerts only". Nothing here moves money --
// this page only reads and marks notifications, both already built on the
// server (`/api/notifications`, `.../read`, `.../read-all`).

import { useCallback, useEffect, useState } from "react";
import { api, type NotificationsState } from "../api";
import { ErrorLine, Working } from "../components/Callout";
import { message, when } from "../util";

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
            <div className="row end mb-4">
              <button className="btn secondary small" onClick={markAllRead} disabled={state.unread_count === 0}>
                Mark all as read
              </button>
            </div>
          )}
          {state.items.length === 0 ? (
            <div className="panel empty">
              <h2>No alerts yet</h2>
              <p className="muted mb-5">
                When a one-off alert fires, or a strategy running as alerts only spots a signal, it lands
                here. Nothing is traded.
              </p>
              <a className="btn secondary no-underline" href="#/orders">
                Set up a one-off alert
              </a>
            </div>
          ) : (
            <ul className="log tall">
              {state.items.map((n) => {
                const content = (
                  <>
                    <time>{when(n.when)}</time>
                    <span>
                      {!n.read && <span className="tag mr-2">unread</span>}
                      <span className={n.read ? undefined : "emph"}>{n.text}</span>
                    </span>
                  </>
                );
                return (
                  <li key={n.id} className={n.read ? undefined : "clickable"}>
                    {n.read ? (
                      content
                    ) : (
                      <button type="button" className="row-button" onClick={() => markRead(n.id)}>
                        {content}
                      </button>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
