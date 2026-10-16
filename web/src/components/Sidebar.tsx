// The menu: a rail down the left on a computer; on a phone, a slim bar at the
// top and the main pages along the bottom, with the rest under "More".

import { useEffect, useState } from "react";
import { chooseTheme, type Theme } from "../theme";
import { Glyph, Icon, type IconName } from "./Icon";

export type NavPage = { id: string; label: string; icon: IconName; phone?: string };

export function Sidebar({
  pages,
  current,
  runningCount,
  unreadCount,
  onRunning,
  onLogout,
}: {
  pages: readonly NavPage[];
  current: string;
  runningCount: number;
  unreadCount: number;
  onRunning: () => void;
  onLogout: () => void;
}) {
  const [theme, setTheme] = useState<Theme>(() => (document.documentElement.dataset.theme as Theme) ?? "light");
  const [moreOpen, setMoreOpen] = useState(false);

  useEffect(() => setMoreOpen(false), [current]);
  useEffect(() => {
    if (!moreOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setMoreOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [moreOpen]);

  const flip = () => {
    const next = theme === "dark" ? "light" : "dark";
    chooseTheme(next);
    setTheme(next);
  };

  const link = (p: NavPage, label = p.label) => (
    <a key={p.id} className="rail-link" href={`#/${p.id}`} aria-current={current === p.id ? "page" : undefined}>
      <Icon name={p.icon} />
      <span>{label}</span>
    </a>
  );

  const running = (
    <button type="button" className="rail-link" onClick={onRunning}>
      <Icon name="running" />
      <span>Running now</span>
      <span className={`rail-count${runningCount > 0 ? " live" : ""}`}>{runningCount}</span>
    </button>
  );

  const alertLabel = unreadCount > 0 ? `Alerts, ${unreadCount} unread` : "Alerts";
  const alerts = (
    <a className="rail-link" href="#/alerts" aria-label={alertLabel} aria-current={current === "alerts" ? "page" : undefined}>
      <Icon name="bell" />
      <span>Alerts</span>
      {unreadCount > 0 && <span className="rail-count alert">{unreadCount > 99 ? "99+" : unreadCount}</span>}
    </a>
  );

  const themeButton = (
    <button type="button" className="rail-link" onClick={flip}>
      <Icon name={theme === "dark" ? "sun" : "moon"} />
      <span>{theme === "dark" ? "Light mode" : "Dark mode"}</span>
    </button>
  );

  const logout = (
    <button type="button" className="rail-link" onClick={onLogout}>
      <Icon name="logout" />
      <span>Log out</span>
    </button>
  );

  const extra = pages.filter((p) => !p.phone);

  return (
    <>
      <nav className="rail" aria-label="Main">
        <a className="wordmark" href="#/build">
          <Glyph />
          Strategy Desk
        </a>
        <div className="rail-group">{pages.map((p) => link(p))}</div>
        <div className="rail-spacer" />
        <div className="rail-group">
          {running}
          {alerts}
          {link({ id: "safety", label: "Safety", icon: "safety" })}
          {themeButton}
          {logout}
        </div>
      </nav>

      <header className="phone-bar">
        <a className="wordmark" href="#/build">
          <Glyph />
          Strategy Desk
        </a>
        <button type="button" className="rail-link" onClick={onRunning} aria-label={`Running now: ${runningCount}`}>
          <Icon name="running" />
          <span className={`rail-count${runningCount > 0 ? " live" : ""}`}>{runningCount}</span>
        </button>
        <a className="rail-link" href="#/alerts" aria-label={alertLabel}>
          <Icon name="bell" />
          {unreadCount > 0 && <span className="rail-count alert">{unreadCount > 99 ? "99+" : unreadCount}</span>}
        </a>
      </header>

      <nav className="tabbar" aria-label="Main">
        {pages.filter((p) => p.phone).map((p) => link(p, p.phone))}
        <button
          type="button"
          className="rail-link"
          aria-expanded={moreOpen}
          aria-current={extra.some((p) => p.id === current) || current === "safety" ? "page" : undefined}
          onClick={() => setMoreOpen(true)}
        >
          <Icon name="more" />
          <span>More</span>
        </button>
      </nav>

      {moreOpen && (
        <>
          <div className="modal-backdrop" onClick={() => setMoreOpen(false)} />
          <div className="more-sheet" role="dialog" aria-label="More pages">
            <div className="rail-group">
              {extra.map((p) => link(p))}
              {link({ id: "safety", label: "Safety", icon: "safety" })}
              {themeButton}
              {logout}
            </div>
          </div>
        </>
      )}
    </>
  );
}
