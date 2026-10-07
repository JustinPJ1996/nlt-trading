import { useCallback, useEffect, useState } from "react";
import { api, whenLoggedOut, type Options, type Result } from "./api";
import { Working } from "./components/Callout";
import { Build, emptyBuild, type BuildState } from "./pages/Build";
import { Login } from "./pages/Login";
import { Paper } from "./pages/Paper";
import { Results } from "./pages/Results";
import { Safety } from "./pages/Safety";
import { Strategies } from "./pages/Strategies";

const PAGES = [
  { id: "build", label: "Build" },
  { id: "results", label: "Results" },
  { id: "strategies", label: "My strategies" },
  { id: "paper", label: "Paper trading" },
  { id: "safety", label: "Safety" },
] as const;

type PageId = (typeof PAGES)[number]["id"];

function pageFromHash(): PageId {
  const id = window.location.hash.replace(/^#\/?/, "");
  return (PAGES.find((p) => p.id === id)?.id ?? "build") as PageId;
}

export function App() {
  const [user, setUser] = useState<string | null | undefined>(undefined);
  const [page, setPage] = useState<PageId>(pageFromHash);
  const [options, setOptions] = useState<Options | null>(null);
  const [build, setBuild] = useState<BuildState>(emptyBuild);
  const [result, setResult] = useState<Result | null>(null);
  const [killed, setKilled] = useState(false);

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

  useEffect(() => {
    if (!user) return;
    api.options().then(setOptions, () => {});
    refreshKill();
  }, [user, refreshKill]);

  useEffect(() => {
    document.title = `${PAGES.find((p) => p.id === page)?.label} - Strategy Desk`;
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
                },
                fromSaved: true,
              });
              go("build");
            }}
            onBuild={() => go("build")}
          />
        )}
        {page === "paper" && <Paper onBuild={() => go("strategies")} />}
        {page === "safety" && <Safety onChange={(s) => setKilled(s.live || s.paper)} />}
      </main>
    </>
  );
}
