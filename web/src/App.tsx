import { useCallback, useEffect, useState } from "react";
import { api, whenLoggedOut, type Options, type Result } from "./api";
import { Working } from "./components/Callout";
import { RunningDrawer } from "./components/RunningDrawer";
import { Sidebar, type NavPage } from "./components/Sidebar";
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
  { id: "build", label: "Build", icon: "build", phone: "Build" },
  { id: "results", label: "Results", icon: "results", phone: "Results" },
  { id: "strategies", label: "My strategies", icon: "strategies", phone: "Saved" },
  { id: "paper", label: "Paper trading", icon: "paper", phone: "Paper" },
  { id: "orders", label: "One-off orders", icon: "orders" },
  { id: "activity", label: "Activity", icon: "activity" },
] as const satisfies readonly NavPage[];

type PageId = (typeof PAGES)[number]["id"] | "alerts" | "safety";

function pageFromHash(): PageId {
  const id = window.location.hash.replace(/^#\/?/, "");
  if (id === "alerts" || id === "safety") return id;
  return (PAGES.find((p) => p.id === id)?.id ?? "build") as PageId;
}

function pageTitle(page: PageId): string {
  if (page === "alerts") return "Alerts";
  if (page === "safety") return "Safety";
  return PAGES.find((p) => p.id === page)?.label ?? "";
}

// What is on the Build and Results pages survives a reload of the tab (but not
// closing it), so a refresh never throws away a strategy being checked.
const SAVED_BUILD = "sd-build";
const SAVED_RESULT = "sd-result";

function remembered<T>(key: string): T | null {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

function remember(key: string, value: unknown): void {
  try {
    if (value == null) sessionStorage.removeItem(key);
    else sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage full or blocked: the page still works, it just forgets on reload.
  }
}

export function App() {
  const [user, setUser] = useState<string | null | undefined>(undefined);
  const [page, setPage] = useState<PageId>(pageFromHash);
  const [options, setOptions] = useState<Options | null>(null);
  const [build, setBuild] = useState<BuildState>(() => remembered<BuildState>(SAVED_BUILD) ?? emptyBuild());
  const [result, setResult] = useState<Result | null>(() => remembered<Result>(SAVED_RESULT));
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

  useEffect(() => remember(SAVED_BUILD, build), [build]);
  useEffect(() => remember(SAVED_RESULT, result), [result]);

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
    <div className="shell">
      <Sidebar
        pages={PAGES}
        current={page}
        runningCount={runningCount}
        unreadCount={unreadCount}
        onRunning={() => setDrawerOpen(true)}
        onLogout={() => {
          api.logout().finally(() => {
            remember(SAVED_BUILD, null);
            remember(SAVED_RESULT, null);
            setUser(null);
          });
        }}
      />
      <div className="content">
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
            onPaper={() => go("paper")}
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
      </div>
    </div>
  );
}
