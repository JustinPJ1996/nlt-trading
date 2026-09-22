-- Strategy versions are immutable and content-addressed.
--
-- The reason is the proving gate: "this strategy passed a backtest and two weeks
-- of paper trading" is only a meaningful statement if the rules cannot have
-- changed since. Editing a strategy therefore creates a new version with a new
-- hash and an unproven state, rather than mutating the row that earned approval.

CREATE TABLE IF NOT EXISTS strategy (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    version         INTEGER NOT NULL,
    spec_hash       TEXT    NOT NULL UNIQUE,
    spec_json       TEXT    NOT NULL,
    description     TEXT    NOT NULL,
    readback        TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    state           TEXT    NOT NULL DEFAULT 'draft',
    allow_unproven  INTEGER NOT NULL DEFAULT 0,
    archived        INTEGER NOT NULL DEFAULT 0,
    UNIQUE (name, version),
    CHECK (state IN ('draft', 'backtested', 'paper', 'live', 'halted'))
);

CREATE TABLE IF NOT EXISTS run (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id     INTEGER NOT NULL REFERENCES strategy(id),
    mode            TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    status          TEXT    NOT NULL DEFAULT 'running',
    params_json     TEXT    NOT NULL DEFAULT '{}',
    metrics_json    TEXT,
    error           TEXT,
    CHECK (mode IN ('backtest', 'paper', 'live')),
    CHECK (status IN ('running', 'complete', 'failed', 'stopped'))
);

CREATE INDEX IF NOT EXISTS idx_run_strategy ON run(strategy_id, mode);

CREATE TABLE IF NOT EXISTS trade (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES run(id),
    symbol          TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    quantity        INTEGER NOT NULL,
    entry_time      TEXT    NOT NULL,
    entry_price     REAL    NOT NULL,
    exit_time       TEXT,
    exit_price      REAL,
    gross_pnl       REAL,
    charges         REAL,
    net_pnl         REAL,
    exit_reason     TEXT,
    entry_reason    TEXT
);

CREATE INDEX IF NOT EXISTS idx_trade_run ON trade(run_id);

-- Every order we ever send, in any mode. Written BEFORE the broker call so a
-- crash mid-flight still leaves evidence that an order may exist.
CREATE TABLE IF NOT EXISTS order_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            INTEGER REFERENCES run(id),
    strategy_id       INTEGER REFERENCES strategy(id),
    client_order_id   TEXT    NOT NULL UNIQUE,
    broker_order_id   TEXT,
    placed_at         TEXT    NOT NULL,
    symbol            TEXT    NOT NULL,
    side              TEXT    NOT NULL,
    quantity          INTEGER NOT NULL,
    order_type        TEXT    NOT NULL,
    limit_price       REAL,
    status            TEXT    NOT NULL,
    filled_quantity   INTEGER NOT NULL DEFAULT 0,
    average_price     REAL,
    rejection_reason  TEXT,
    CHECK (side IN ('buy', 'sell'))
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    trade_date      TEXT    NOT NULL,
    strategy_id     INTEGER NOT NULL REFERENCES strategy(id),
    mode            TEXT    NOT NULL,
    realised_pnl    REAL    NOT NULL DEFAULT 0,
    charges         REAL    NOT NULL DEFAULT 0,
    trade_count     INTEGER NOT NULL DEFAULT 0,
    halted          INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (trade_date, strategy_id, mode)
);

-- Append-only. Answers "why did it buy that?" months later, which is the only
-- way to debug an automated system that trades while nobody is watching.
CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    at              TEXT    NOT NULL,
    strategy_id     INTEGER REFERENCES strategy(id),
    run_id          INTEGER REFERENCES run(id),
    event           TEXT    NOT NULL,
    detail_json     TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_at ON audit_log(at);

-- Single-row table holding the kill switch. A row rather than a file so that
-- reading it is transactional with everything else the risk manager checks.
CREATE TABLE IF NOT EXISTS kill_switch (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    engaged         INTEGER NOT NULL DEFAULT 0,
    engaged_at      TEXT,
    reason          TEXT
);

INSERT OR IGNORE INTO kill_switch (id, engaged) VALUES (1, 0);
