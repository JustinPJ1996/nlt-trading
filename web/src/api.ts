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
    throw new ApiError(
      data?.error ?? "Something went wrong on our side. Try again in a moment.",
      res.status,
    );
  }
  return data as T;
}

// ------------------------------------------------------------------- shapes

// Where a readback line came from. Null when unknown (a saved strategy).
export type ReadbackLabel = "said" | "worked_out" | "default";

export type StrategySettings = {
  algo_type: "intraday" | "multi_day";
  // False on daily candles: those are always multi-day.
  can_be_intraday: boolean;
  entry_cutoff: boolean;
  entry_cutoff_minutes: number;
  square_off_minutes: number;
  close: string;
  close_note: string;
  limits: { entry_cutoff_minutes: [number, number]; square_off_minutes: [number, number] };
};

// What the Settings panel can change; sent to the server as edits.
export type SettingsEdits = Partial<{
  algo_type: "intraday" | "multi_day";
  entry_cutoff: boolean;
  entry_cutoff_minutes: number;
  square_off_minutes: number;
}>;

export type SettingsSaved = {
  id: number;
  version: number;
  new_version: boolean;
  readback: string;
  settings: StrategySettings;
  notes: string[];
  problem: string | null;
};

export type Understood = {
  settings?: StrategySettings;
  spec: unknown;
  readback: string;
  // One per body line of `readback`, in order; null when not known.
  readback_labels?: (ReadbackLabel | null)[] | null;
  // Where each part came from, kept so form edits can keep the labels.
  sources?: Record<string, ReadbackLabel> | null;
  name: string;
  symbol: string;
  timeframe: string;
  // Candle sizes the Candles box may offer; null/absent means all of them.
  allowed_timeframes?: string[] | null;
  // Why this can only be paper traded (a strike the user named); null otherwise.
  backtest_problem?: string | null;
  // Boxes for the numbers inside the buy/sell rule: levels and lengths.
  rule_boxes?: RuleBox[];
};

export type RuleBox = { key: string; group: string; label: string; value: number; hint: string };

export type Choice = { label: string; value: string };

export type Question = {
  field: string;
  text: string;
  why: string;
  suggestion: string | null;
  // Tap-to-answer buttons. None is ever chosen for the user.
  choices: Choice[];
  // Whether a typed answer is read. With no choices either, the only way to
  // answer is to change the description.
  answerable: boolean;
};

export type Translation = {
  understood: Understood | null;
  questions: Question[];
  unparsed: string[];
  notes: string[];
  read_by_ai: boolean;
  answers: Record<string, string>;
  // The test period the sentence asked for, already turned into dates.
  period: { words: string; start: string; end: string; note: string | null } | null;
};

// A ready-made strategy with its own tested numbers (recomputed daily).
export type LibraryCard = {
  title: string;
  sentence: string;
  error?: string;
  name?: string;
  return?: string;
  holding_return?: string;
  benchmark_name?: string;
  win_rate?: string;
  worst_drop?: string;
  trades?: number;
  first?: string;
  last?: string;
  serious_warning?: boolean;
  // % gained on the starting money, strategy (s) and simply holding (b), thinned.
  curve?: { s: number[]; b: number[] };
};

export type Library = {
  tested_to: string;
  capital_text: string;
  // How many simple strategies were tried to choose these (said on the page).
  tried?: number;
  cards: LibraryCard[];
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
  // Options only: why this contract was chosen, in words.
  why: string | null;
  pnl: number;
  pnl_text: string;
  // Sold and multi-leg option positions: every option inside this one row.
  legs?: TradeLeg[];
};

export type TradeLeg = {
  label: string;
  side: string;
  quantity: number;
  entered: string;
  entry_price: string;
  exited: string;
  exit_price: string;
  stop: string | null;
  reason: string;
  reentry: number;
  why: string;
  pnl: number;
  pnl_text: string;
};

// A run's slippage setting: either a percentage of the price or a fixed
// rupee amount per unit. Zero (the default everywhere) means no slippage is
// assumed at all -- see `slippage_text` on `BacktestParams` for how that is
// said out loud.
export type Slippage = { kind: "pct" | "rs"; value: number };

export const ZERO_SLIPPAGE: Slippage = { kind: "pct", value: 0 };

export type BacktestParams = {
  capital: number;
  capital_text: string;
  cost_model: string;
  charges_applied?: string;
  slippage: Slippage;
  slippage_text: string;
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
  // The same strategy run again with charges switched off. Null when no
  // charges were paid, so there is nothing to compare.
  before_after_charges: { label: string; before: string; after: string }[] | null;
  // How the test was done (plain words, from the engine's own rules).
  assumptions: string[];
  weak_spots: string[];
  all_metrics: Record<string, number | null>;
  chart: ChartPoint[];
  trades: Trade[];
  notes: string[];
  // The test covered different dates from the ones asked for (option prices
  // exist only for some); said at the very top.
  window_notes: string[];
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
  paused: boolean;
  health: "success" | "info" | "warning" | "error";
  health_label: string;
  message: string;
  started: string;
  // True for a run started with "Start as alerts only": it trades nothing,
  // so capital_text is "" and snapshot is always null.
  alert_only: boolean;
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
  events: { when: string; text: string; drift: boolean; trade_key: string | null }[];
};

export type PaperState = {
  kill_switch: boolean;
  job_installed: boolean;
  kite: { state: string; line: string; has_login: boolean };
  runs: PaperRun[];
  startable: {
    id: number;
    name: string;
    version: number;
    readback: string;
    problem: string | null;
    notes: string[];
    settings?: StrategySettings;
  }[];
  cost_models: string[];
  default_capital: number;
};

export type SafetyState = {
  live: boolean;
  paper: boolean;
  freshness: { symbol: string; level: "good" | "critical" | "info"; text: string }[];
  audit: { when: string; event: string; detail: string }[];
};

export type OrderPosition = {
  id: number;
  name: string;
  what: string;
  entered: string;
  entry_price: string;
  last_price: string;
  last_at: string;
  stop: string;
  target: string;
  pnl: number | null;
  pnl_text: string;
  expiry: string;
};

export type OrderReading = {
  order: unknown | null;
  readback: string[];
  warnings: string[];
  notes: string[];
  questions: Question[];
  answers: Record<string, string>;
  offer_stop: boolean;
  needs_positions: boolean;
  positions: OrderPosition[];
};

export type SavedOrder = {
  id: number;
  sentence: string;
  readback: string[];
  saved: string;
  status: "waiting" | "filling" | "open" | "done" | "cancelled" | "alerted";
  status_label: string;
  can_cancel: boolean;
  result: number | null;
  result_text: string;
  events: { when: string; text: string }[];
};

export type OrdersState = {
  kill_switch: boolean;
  job_installed: boolean;
  checker: { health: string; message: string; checked: string };
  positions: OrderPosition[];
  orders: SavedOrder[];
};

// One merged, newest-first row across every paper run and every one-off
// order -- what `/api/activity` returns.
export type ActivityItem = {
  time: string;
  source: "strategy" | "order";
  source_name: string;
  source_id: number;
  kind: string;
  text: string;
  skip: boolean;
  trade_key: string | null;
};

export type ActivityState = { items: ActivityItem[] };

export type ActivityFilter = { source_id?: number; skips_only?: boolean; limit?: number };

// The header "Running" drawer: every strategy run, every waiting one-off
// order, and every open one-off position, in one list.
export type RunningItem = {
  kind: "strategy" | "one_off_order" | "one_off_position";
  run_id?: number;
  order_id?: number;
  position_id?: number;
  name: string;
  health: "success" | "info" | "warning" | "error";
  health_label: string;
  paused: boolean;
  // True only for a "strategy" item started as alerts only: open_now and
  // total read "Alerts only" instead of a rupee figure, and are never part
  // of the drawer's totals.
  alert_only?: boolean;
  closed_today: string;
  closed_today_value: number;
  open_now: string;
  open_now_value: number;
  open_now_as_of: string;
  total: string;
  total_value: number;
};

export type RunningState = {
  items: RunningItem[];
  totals: { closed_today: string; open_now: string; total: string; label: string };
};

// The inbox: a one-off alert firing, or an alert-only strategy's signal/exit.
export type Notification = {
  id: number;
  when: string;
  source: "one_off_alert" | "strategy_alert";
  source_id: number;
  text: string;
  read: boolean;
};

export type NotificationsState = { unread_count: number; items: Notification[] };

// -------------------------------------------------------------------- calls

export const api = {
  me: () => call<{ user: string }>("/api/me"),
  login: (login_id: string, password: string) =>
    call<{ user: string }>("/api/login", { login_id, password }),
  logout: () => call<{ ok: boolean }>("/api/logout", {}),
  options: () => call<Options>("/api/options"),
  translate: (text: string, answers: Record<string, string>) =>
    call<Translation>("/api/translate", { text, answers }),
  library: () => call<Library>("/api/library"),
  adjust: (body: {
    spec: unknown;
    sources: Record<string, ReadbackLabel> | null | undefined;
    answers: Record<string, string>;
    edits: Record<string, string | number | boolean>;
  }) => call<Understood>("/api/adjust", body),
  backtest: (body: {
    spec: unknown;
    capital: number;
    cost_model: string;
    start: string;
    end: string;
    slippage?: Slippage;
  }) => call<Result>("/api/backtest", body),
  save: (result_id: string) =>
    call<{ strategy_id: number; name: string }>("/api/strategies", { result_id }),
  saveUntested: (spec: unknown) =>
    call<{ strategy_id: number; name: string }>("/api/strategies/untested", { spec }),
  strategies: () => call<{ strategies: SavedStrategy[] }>("/api/strategies"),
  savedSpec: (id: number) =>
    call<{ description: string; understood: Understood }>(`/api/strategies/${id}/spec`),
  saveSettings: (id: number, edits: SettingsEdits) =>
    call<SettingsSaved>(`/api/strategies/${id}/settings`, { edits }),
  paper: () => call<PaperState>("/api/paper"),
  paperStart: (
    strategy_id: number,
    capital: number,
    cost_model: string,
    alert_only = false,
    slippage?: Slippage,
  ) =>
    call<{ run_id: number }>("/api/paper/start", {
      strategy_id,
      capital,
      cost_model,
      alert_only,
      slippage,
    }),
  paperCheck: (id: number) => call<{ message: string }>(`/api/paper/${id}/check`, {}),
  paperStop: (id: number) => call<{ ok: boolean }>(`/api/paper/${id}/stop`, {}),
  paperPause: (id: number) => call<{ run: PaperRun }>(`/api/paper/${id}/pause`, {}),
  paperResume: (id: number) => call<{ run: PaperRun }>(`/api/paper/${id}/resume`, {}),
  running: () => call<RunningState>("/api/running"),
  kiteLogin: (user_id: string, password: string, secret: string) =>
    call<{ ok: boolean }>("/api/kite/login", { user_id, password, secret }),
  kiteForget: () => call<{ ok: boolean }>("/api/kite/forget", {}),
  kiteToken: (token: string) => call<{ ok: boolean }>("/api/kite/token", { token }),
  orders: () => call<OrdersState>("/api/orders"),
  orderRead: (text: string, answers: Record<string, string>) =>
    call<OrderReading>("/api/orders/read", { text, answers }),
  orderCreate: (
    text: string,
    answers: Record<string, string>,
    order: unknown,
    positions: number[],
    slippage?: Slippage,
  ) =>
    call<{ order_id: number }>("/api/orders/create", { text, answers, order, positions, slippage }),
  orderCancel: (id: number) => call<{ ok: boolean }>(`/api/orders/${id}/cancel`, {}),
  orderClose: (positionId: number) =>
    call<{ order_id: number }>(`/api/orders/positions/${positionId}/close`, {}),
  ordersCheck: () => call<{ message: string }>("/api/orders/check", {}),
  safety: () => call<SafetyState>("/api/safety"),
  killSwitch: (which: "live" | "paper", action: "engage" | "release", confirm = false) =>
    call<SafetyState>(`/api/safety/${which}/${action}`, { confirm }),
  activity: (filter: ActivityFilter = {}) => {
    const q = new URLSearchParams();
    if (filter.source_id !== undefined) q.set("source_id", String(filter.source_id));
    if (filter.skips_only) q.set("skips_only", "1");
    if (filter.limit !== undefined) q.set("limit", String(filter.limit));
    const qs = q.toString();
    return call<ActivityState>(`/api/activity${qs ? `?${qs}` : ""}`);
  },
  paperTrade: (runId: number, tradeKey: string) =>
    call<{ steps: string[] }>(`/api/paper/${runId}/trade/${encodeURIComponent(tradeKey)}`),
  orderTimeline: (orderId: number) => call<{ steps: string[] }>(`/api/orders/${orderId}/timeline`),
  notifications: () => call<NotificationsState>("/api/notifications"),
  notificationRead: (id: number) => call<NotificationsState>(`/api/notifications/${id}/read`, {}),
  notificationReadAll: () => call<NotificationsState>("/api/notifications/read-all", {}),
};
