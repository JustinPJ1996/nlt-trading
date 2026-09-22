"""Persistence.

Plain sqlite3 rather than an ORM: the schema is small, the queries are simple,
and a single file with no migration machinery is the right weight for something
one person operates. The schema lives in `schema.sql` next to this module.

Two invariants the rest of the system leans on:

* A strategy version is content-addressed by the hash of its spec, so approval
  earned by one set of rules cannot silently transfer to different rules.
* `audit_log` is append-only. Nothing in this module updates or deletes from it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from nlt.spec.models import StrategySpec

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "nlt.sqlite"
SCHEMA = Path(__file__).resolve().parent / "schema.sql"


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def spec_hash(spec: StrategySpec) -> str:
    """Stable content hash of a spec.

    `sort_keys` matters: the same strategy must hash identically regardless of
    the field order Claude happened to emit.
    """
    payload = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Store:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        # WAL lets the dashboard read while a runner writes, which is the normal
        # operating shape once paper trading runs in the background.
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(SCHEMA.read_text())
        finally:
            conn.close()

    # ------------------------------------------------------------ strategies

    def save_strategy(self, spec: StrategySpec, readback: str) -> int:
        """Insert a new strategy version, or return the existing id if unchanged.

        Saving the same spec twice is a no-op rather than an error -- the user
        re-running a translation that produced identical rules should not create
        a second version and lose the first one's proving history.
        """
        digest = spec_hash(spec)

        with self.tx() as conn:
            existing = conn.execute(
                "SELECT id FROM strategy WHERE spec_hash = ?", (digest,)
            ).fetchone()
            if existing:
                return int(existing["id"])

            version = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 AS v FROM strategy WHERE name = ?",
                (spec.name,),
            ).fetchone()["v"]

            cur = conn.execute(
                """INSERT INTO strategy
                   (name, version, spec_hash, spec_json, description, readback, created_at, state)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'draft')""",
                (
                    spec.name,
                    version,
                    digest,
                    json.dumps(spec.model_dump(mode="json")),
                    spec.description,
                    readback,
                    now(),
                ),
            )
            strategy_id = int(cur.lastrowid)
            self._audit(conn, "strategy_created", strategy_id, None,
                        {"name": spec.name, "version": version, "hash": digest})
            return strategy_id

    def get_strategy(self, strategy_id: int) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM strategy WHERE id = ?", (strategy_id,)).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def load_spec(self, strategy_id: int) -> StrategySpec:
        row = self.get_strategy(strategy_id)
        if row is None:
            raise KeyError(f"no strategy with id {strategy_id}")
        return StrategySpec.model_validate(json.loads(row["spec_json"]))

    def list_strategies(self, include_archived: bool = False) -> list[dict]:
        sql = "SELECT * FROM strategy"
        if not include_archived:
            sql += " WHERE archived = 0"
        sql += " ORDER BY name, version DESC"
        conn = self._connect()
        try:
            return [dict(r) for r in conn.execute(sql)]
        finally:
            conn.close()

    def set_state(self, strategy_id: int, state: str) -> None:
        """Advance a strategy's lifecycle state, refusing to skip the proving gate."""
        if state == "live" and not self.is_proven(strategy_id):
            raise PermissionError(
                "this strategy has not been paper traded yet. Run it in paper mode "
                "first, or set allow_unproven on it to deliberately skip that step."
            )
        with self.tx() as conn:
            conn.execute("UPDATE strategy SET state = ? WHERE id = ?", (state, strategy_id))
            self._audit(conn, "state_changed", strategy_id, None, {"state": state})

    def is_proven(self, strategy_id: int) -> bool:
        """Has this exact spec earned the right to trade real money?

        Requires a completed backtest AND a completed paper run. The override is
        explicit and recorded, so skipping the gate is a decision rather than an
        accident.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT allow_unproven FROM strategy WHERE id = ?", (strategy_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"no strategy with id {strategy_id}")
            if row["allow_unproven"]:
                return True

            modes = {
                r["mode"]
                for r in conn.execute(
                    "SELECT DISTINCT mode FROM run WHERE strategy_id = ? AND status = 'complete'",
                    (strategy_id,),
                )
            }
        finally:
            conn.close()
        return {"backtest", "paper"} <= modes

    def set_allow_unproven(self, strategy_id: int, allow: bool, reason: str) -> None:
        with self.tx() as conn:
            conn.execute(
                "UPDATE strategy SET allow_unproven = ? WHERE id = ?", (int(allow), strategy_id)
            )
            self._audit(conn, "proving_gate_override", strategy_id, None,
                        {"allow_unproven": allow, "reason": reason})

    # ------------------------------------------------------------------ runs

    def start_run(self, strategy_id: int, mode: str, params: dict | None = None) -> int:
        with self.tx() as conn:
            cur = conn.execute(
                "INSERT INTO run (strategy_id, mode, started_at, params_json) VALUES (?, ?, ?, ?)",
                (strategy_id, mode, now(), json.dumps(params or {})),
            )
            run_id = int(cur.lastrowid)
            self._audit(conn, "run_started", strategy_id, run_id, {"mode": mode})
            return run_id

    def finish_run(self, run_id: int, metrics: dict, status: str = "complete") -> None:
        with self.tx() as conn:
            conn.execute(
                "UPDATE run SET finished_at = ?, status = ?, metrics_json = ? WHERE id = ?",
                (now(), status, json.dumps(metrics, default=str), run_id),
            )
            self._audit(conn, "run_finished", None, run_id, {"status": status})

    def fail_run(self, run_id: int, error: str) -> None:
        with self.tx() as conn:
            conn.execute(
                "UPDATE run SET finished_at = ?, status = 'failed', error = ? WHERE id = ?",
                (now(), error, run_id),
            )
            self._audit(conn, "run_failed", None, run_id, {"error": error[:500]})

    def list_runs(self, strategy_id: int | None = None, mode: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM run WHERE 1=1", []
        if strategy_id is not None:
            sql, args = sql + " AND strategy_id = ?", args + [strategy_id]
        if mode is not None:
            sql, args = sql + " AND mode = ?", args + [mode]
        conn = self._connect()
        try:
            return [dict(r) for r in conn.execute(sql + " ORDER BY started_at DESC", args)]
        finally:
            conn.close()

    def save_trades(self, run_id: int, trades: list, symbol: str = "") -> None:
        """Persist trades from a completed run. `trades` are engine Trade objects."""
        rows = [
            (
                run_id,
                getattr(t, "symbol", symbol),
                t.direction,
                t.quantity,
                str(t.entry_time),
                t.entry_price,
                str(t.exit_time),
                t.exit_price,
                t.gross_pnl,
                t.charges,
                t.net_pnl,
                t.exit_reason,
                getattr(t, "entry_reason", ""),
            )
            for t in trades
        ]
        with self.tx() as conn:
            conn.executemany(
                """INSERT INTO trade (run_id, symbol, direction, quantity, entry_time,
                   entry_price, exit_time, exit_price, gross_pnl, charges, net_pnl,
                   exit_reason, entry_reason)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    def trades(self, run_id: int) -> list[dict]:
        conn = self._connect()
        try:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM trade WHERE run_id = ? ORDER BY entry_time", (run_id,)
                )
            ]
        finally:
            conn.close()

    # ---------------------------------------------------------- kill switch

    def kill_switch_engaged(self) -> bool:
        conn = self._connect()
        try:
            return bool(conn.execute("SELECT engaged FROM kill_switch WHERE id = 1").fetchone()[0])
        finally:
            conn.close()

    def engage_kill_switch(self, reason: str) -> None:
        with self.tx() as conn:
            conn.execute(
                "UPDATE kill_switch SET engaged = 1, engaged_at = ?, reason = ? WHERE id = 1",
                (now(), reason),
            )
            conn.execute("UPDATE strategy SET state = 'halted' WHERE state = 'live'")
            self._audit(conn, "kill_switch_engaged", None, None, {"reason": reason})

    def release_kill_switch(self, reason: str) -> None:
        """Deliberately does NOT re-arm anything.

        Strategies halted by the kill switch stay halted and must be armed again
        one at a time. An operator hitting stop during a bad day should not find
        everything trading again because they cleared the flag.
        """
        with self.tx() as conn:
            conn.execute(
                "UPDATE kill_switch SET engaged = 0, engaged_at = NULL, reason = NULL WHERE id = 1"
            )
            self._audit(conn, "kill_switch_released", None, None, {"reason": reason})

    # ----------------------------------------------------------- daily pnl

    def record_daily_pnl(
        self, trade_date: str, strategy_id: int, mode: str, pnl: float, charges: float
    ) -> None:
        with self.tx() as conn:
            conn.execute(
                """INSERT INTO daily_pnl (trade_date, strategy_id, mode, realised_pnl,
                   charges, trade_count)
                   VALUES (?, ?, ?, ?, ?, 1)
                   ON CONFLICT(trade_date, strategy_id, mode) DO UPDATE SET
                     realised_pnl = realised_pnl + excluded.realised_pnl,
                     charges      = charges + excluded.charges,
                     trade_count  = trade_count + 1""",
                (trade_date, strategy_id, mode, pnl, charges),
            )

    def daily_pnl(self, trade_date: str, strategy_id: int, mode: str) -> float:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT realised_pnl FROM daily_pnl WHERE trade_date = ? AND strategy_id = ? "
                "AND mode = ?",
                (trade_date, strategy_id, mode),
            ).fetchone()
        finally:
            conn.close()
        return float(row["realised_pnl"]) if row else 0.0

    # --------------------------------------------------------------- audit

    def _audit(self, conn, event: str, strategy_id, run_id, detail: dict) -> None:
        conn.execute(
            "INSERT INTO audit_log (at, strategy_id, run_id, event, detail_json) VALUES (?,?,?,?,?)",
            (now(), strategy_id, run_id, event, json.dumps(detail, default=str)),
        )

    def audit(self, event: str, strategy_id=None, run_id=None, **detail) -> None:
        with self.tx() as conn:
            self._audit(conn, event, strategy_id, run_id, detail)

    def audit_trail(self, limit: int = 200) -> list[dict]:
        conn = self._connect()
        try:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
                )
            ]
        finally:
            conn.close()
