// Every call to the Python side goes through here. The server does all the
// thinking -- reading the sentence, writing the readback, running the engine,
// formatting rupees -- and this file only carries requests and answers.

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

let onLoggedOut: () => void = () => {};
export function whenLoggedOut(fn: () => void) {
  onLoggedOut = fn;
}

async function call<T>(path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: "same-origin",
    });
  } catch {
    throw new ApiError("Could not reach the app. Check your internet connection and try again.", 0);
  }
  let data: any = null;
  try {
    data = await res.json();
  } catch {
    // An empty or non-JSON reply is handled below as a generic failure.
  }
  if (!res.ok) {
    if (res.status === 401 && path !== "/api/login") onLoggedOut();
    throw new ApiError(data?.error ?? `The app answered with an error (${res.status}).`, res.status);
  }
  return data as T;
}

// ------------------------------------------------------------------- shapes

export type Understood = {
  spec: unknown;
  readback: string;
  name: string;
  symbol: string;
  timeframe: string;
};

export type Question = { field: string; text: string; why: string; suggestion: string | null };

export type Translation = {
  understood: Understood | null;
  questions: Question[];
  unparsed: string[];
  notes: string[];
  read_by_ai: boolean;
  answers: Record<string, string>;
};

export type Options = {
  examples: string[];
  cost_models: string[];
  default_capital: number;
  default_start: string;
  default_end: string;
};

export type Flag = { severity: "critical" | "warning" | "info"; headline: string; detail: string };

export type ChartPoint = { t: string; s: number; b: number; sd: number; bd: number };

export type Trade = {
  symbol: string;
  direction: string;
  entered: string;
  entry_price: string;
  exited: string;
  exit_price: string;
  reason: string;
  pnl: number;
  pnl_text: string;
};

export type BacktestParams = {
  capital: number;
  capital_text: string;
  cost_model: string;
  start: string | null;
  end: string | null;
};

export type Result = {
  result_id: string;
  tested: Understood;
  params: BacktestParams;
  verdict: { summary: string; passed: boolean; flags: Flag[] };
  headline: {
    strategy_return: string;
    benchmark_return: string;
    strategy_return_raw: number;
    benchmark_return_raw: number;
    benchmark_name: string;
    first: string;
    last: string;
  };
  metrics: { label: string; strategy: string; benchmark: string }[];
  all_metrics: Record<string, number | null>;
  chart: ChartPoint[];
  trades: Trade[];
  notes: string[];
};

export type RunSummary = {
  id: number;
  kind: string;
  status: string;
  started: string;
  finished: string;
  return_text: string;
  capital_text: string;
  error: string;
};

export type SavedStrategy = {
  id: number;
  name: string;
  version: number;
  state: string;
  proven: boolean;
  created: string;
  description: string;
  readback: string;
  runs: RunSummary[];
};

export type PaperRun = {
  run_id: number;
  name: string;
  version: number;
  readback: string;
  health: "success" | "info" | "warning" | "error";
  health_label: string;
  message: string;
  started: string;
  capital_text: string;
  checked: string;
  daily: boolean;
  snapshot: { equity: string; realised: string; open: string } | null;
  positions: {
    what: string;
    direction: string;
    entry_price: string;
    last_price: string;
    pnl: number | null;
    pnl_text: string;
  }[];
  events: { when: string; text: string; drift: boolean }[];
};

export type PaperState = {
  kill_switch: boolean;
  job_installed: boolean;
  kite: { state: string; line: string; has_login: boolean };
  runs: PaperRun[];
  startable: { id: number; name: string; version: number; readback: string; problem: string | null }[];
  cost_models: string[];
  default_capital: number;
};

export type SafetyState = {
  live: boolean;
  paper: boolean;
  freshness: { symbol: string; level: "good" | "critical" | "info"; text: string }[];
  audit: { when: string; event: string; detail: string }[];
};

// -------------------------------------------------------------------- calls

export const api = {
  me: () => call<{ user: string }>("/api/me"),
  login: (login_id: string, password: string) =>
    call<{ user: string }>("/api/login", { login_id, password }),
  logout: () => call<{ ok: boolean }>("/api/logout", {}),
  options: () => call<Options>("/api/options"),
  translate: (text: string, answers: Record<string, string>) =>
    call<Translation>("/api/translate", { text, answers }),
  backtest: (body: {
    spec: unknown;
    capital: number;
    cost_model: string;
    start: string;
    end: string;
  }) => call<Result>("/api/backtest", body),
  save: (result_id: string) =>
    call<{ strategy_id: number; name: string }>("/api/strategies", { result_id }),
  strategies: () => call<{ strategies: SavedStrategy[] }>("/api/strategies"),
  savedSpec: (id: number) =>
    call<{ description: string; understood: Understood }>(`/api/strategies/${id}/spec`),
  paper: () => call<PaperState>("/api/paper"),
  paperStart: (strategy_id: number, capital: number, cost_model: string) =>
    call<{ run_id: number }>("/api/paper/start", { strategy_id, capital, cost_model }),
  paperCheck: (id: number) => call<{ message: string }>(`/api/paper/${id}/check`, {}),
  paperStop: (id: number) => call<{ ok: boolean }>(`/api/paper/${id}/stop`, {}),
  kiteLogin: (user_id: string, password: string, secret: string) =>
    call<{ ok: boolean }>("/api/kite/login", { user_id, password, secret }),
  kiteForget: () => call<{ ok: boolean }>("/api/kite/forget", {}),
  kiteToken: (token: string) => call<{ ok: boolean }>("/api/kite/token", { token }),
  safety: () => call<SafetyState>("/api/safety"),
  killSwitch: (which: "live" | "paper", action: "engage" | "release", confirm = false) =>
    call<SafetyState>(`/api/safety/${which}/${action}`, { confirm }),
};
