import { useCallback, useEffect, useState } from "react";
import { api, whenLoggedOut, type Options, type Result } from "./api";
import { Working } from "./components/Callout";
import { RunningDrawer } from "./components/RunningDrawer";
import { Activity } from "./pages/Activity";
import { Alerts } from "./pages/Alerts";
import { Build, emptyBuild, type BuildState } from "./pages/Build";
import { Login } from "./pages/Login";
import { Orders } from "./pages/Orders";
import { Paper } from "./pages/Paper";
import { Results } from "./pages/Results";
import { Safety } from "./pages/Safety";
import { Strategies } from "./pages/Strategies";

const PAGES = [
  { id: "build", label: "Build" },
  { id: "results", label: "Results" },
  { id: "strategies", label: "My strategies" },
  { id: "paper", label: "Paper trading" },
  { id: "orders", label: "One-off orders" },
  { id: "activity", label: "Activity" },
  { id: "safety", label: "Safety" },
] as const;

type PageId = (typeof PAGES)[number]["id"] | "alerts";

function pageFromHash(): PageId {
  const id = window.location.hash.replace(/^#\/?/, "");
  if (id === "alerts") return "alerts";
  return (PAGES.find((p) => p.id === id)?.id ?? "build") as PageId;
}

function pageTitle(page: PageId): string {
  if (page === "alerts") return "Alerts";
  return PAGES.find((p) => p.id === page)?.label ?? "";
}

export function App() {
  const [user, setUser] = useState<string | null | undefined>(undefined);
  const [page, setPage] = useState<PageId>(pageFromHash);
  const [options, setOptions] = useState<Options | null>(null);
  const [build, setBuild] = useState<BuildState>(emptyBuild);
  const [result, setResult] = useState<Result | null>(null);
  const [killed, setKilled] = useState(false);
  const [runningCount, setRunningCount] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);

  useEffect(() => {
    whenLoggedOut(() => setUser(null));
    api.me().then((r) => setUser(r.user), () => setUser(null));
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const refreshKill = useCallback(() => {
    api.safety().then((s) => setKilled(s.paper || s.live), () => {});
  }, []);

  const refreshRunningCount = useCallback(() => {
    api.running().then((s) => setRunningCount(s.items.length), () => {});
  }, []);

  const refreshUnreadCount = useCallback(() => {
    api.notifications().then((s) => setUnreadCount(s.unread_count), () => {});
  }, []);

  useEffect(() => {
    if (!user) return;
    api.options().then(setOptions, () => {});
    refreshKill();
    refreshRunningCount();
    refreshUnreadCount();
    const id = window.setInterval(refreshRunningCount, 30_000);
    const id2 = window.setInterval(refreshUnreadCount, 30_000);
    return () => {
      window.clearInterval(id);
      window.clearInterval(id2);
    };
  }, [user, refreshKill, refreshRunningCount, refreshUnreadCount]);

  useEffect(() => {
    document.title = `${pageTitle(page)} - Strategy Desk`;
    window.scrollTo(0, 0);
  }, [page]);

  const go = (id: PageId) => {
    window.location.hash = `/${id}`;
  };

  if (user === undefined) {
    return (
      <main>
        <Working>Opening...</Working>
      </main>
    );
  }
  if (user === null) return <Login onDone={setUser} />;

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <span className="wordmark">Strategy Desk</span>
          <nav className="nav" aria-label="Main">
            {PAGES.map((p) => (
              <a key={p.id} href={`#/${p.id}`} aria-current={page === p.id ? "page" : undefined}>
                {p.label}
              </a>
            ))}
          </nav>
          <button className="btn secondary small" onClick={() => setDrawerOpen(true)}>
            Running &middot; {runningCount}
          </button>
          <button
            className="bell-btn"
            aria-label={unreadCount > 0 ? `Alerts, ${unreadCount} unread` : "Alerts"}
            aria-current={page === "alerts" ? "page" : undefined}
            onClick={() => (window.location.hash = "/alerts")}
          >
            <span aria-hidden="true">&#128276;</span>
            {unreadCount > 0 && <span className="bell-badge">{unreadCount > 99 ? "99+" : unreadCount}</span>}
          </button>
          <button
            className="signout"
            onClick={() => api.logout().finally(() => setUser(null))}
          >
            Log out
          </button>
        </div>
      </header>
      {killed && (
        <div className="killbanner" role="alert">
          A kill switch is on. <a href="#/safety">See Safety</a>
        </div>
      )}
      {drawerOpen && (
        <RunningDrawer
          onClose={() => {
            setDrawerOpen(false);
            refreshRunningCount();
          }}
        />
      )}
      <main>
        {page === "build" && (
          <Build
            options={options}
            state={build}
            setState={setBuild}
            onResult={(r) => {
              setResult(r);
              go("results");
            }}
          />
        )}
        {page === "results" && (
          <Results result={result} onEdit={() => go("build")} onSaved={() => {}} />
        )}
        {page === "strategies" && (
          <Strategies
            onTestAgain={(description, understood) => {
              setBuild({
                ...emptyBuild(),
                ...(options ? { capital: String(options.default_capital), costModel: options.cost_models[0], start: options.default_start, end: options.default_end } : {}),
                text: description,
                checkedText: description,
                translation: {
                  understood,
                  questions: [],
                  unparsed: [],
                  notes: [],
                  read_by_ai: false,
                  answers: {},
                  period: null,
                },
                fromSaved: true,
              });
              go("build");
            }}
            onBuild={() => go("build")}
          />
        )}
        {page === "paper" && <Paper onBuild={() => go("strategies")} />}
        {page === "orders" && <Orders />}
        {page === "activity" && <Activity />}
        {page === "safety" && <Safety onChange={(s) => setKilled(s.live || s.paper)} />}
        {page === "alerts" && <Alerts onChanged={refreshUnreadCount} />}
      </main>
    </>
  );
}
