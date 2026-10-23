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
from collections.abc import Generator
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
    def tx(self) -> Generator[sqlite3.Connection]:
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
            strategy_id = int(cur.lastrowid)  # ty: ignore[invalid-argument-type]
            self._audit(
                conn,
                "strategy_created",
                strategy_id,
                None,
                {"name": spec.name, "version": version, "hash": digest},
            )
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
                    "SELECT DISTINCT mode FROM run WHERE strategy_id = ? AND status = 'complete' "
                    "AND id NOT IN (SELECT run_id FROM paper_run_option WHERE alert_only = 1)",
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
            self._audit(
                conn,
                "proving_gate_override",
                strategy_id,
                None,
                {"allow_unproven": allow, "reason": reason},
            )

    # ------------------------------------------------------------------ runs

    def start_run(
        self, strategy_id: int, mode: str, params: dict | None = None, *, alert_only: bool = False
    ) -> int:
        with self.tx() as conn:
            cur = conn.execute(
                "INSERT INTO run (strategy_id, mode, started_at, params_json) VALUES (?, ?, ?, ?)",
                (strategy_id, mode, now(), json.dumps(params or {})),
            )
            run_id = int(cur.lastrowid)  # ty: ignore[invalid-argument-type]
            if alert_only:
                # Same transaction as the run itself: the every-minute paper job
                # must never see this run before it is marked alert-only, or it
                # would record paper money for it.
                conn.execute(
                    "INSERT INTO paper_run_option (run_id, alert_only) VALUES (?, 1)", (run_id,)
                )
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

    # ---------------------------------------------------- paper kill switch

    def paper_kill_switch_engaged(self) -> bool:
        conn = self._connect()
        try:
            row = conn.execute("SELECT engaged FROM paper_kill_switch WHERE id = 1").fetchone()
            return bool(row[0])
        finally:
            conn.close()

    def engage_paper_kill_switch(self, reason: str) -> None:
        """Pauses every paper run. Touches nothing live, and nothing already recorded."""
        with self.tx() as conn:
            conn.execute(
                "UPDATE paper_kill_switch SET engaged = 1, engaged_at = ?, reason = ? WHERE id = 1",
                (now(), reason),
            )
            self._audit(conn, "paper_kill_switch_engaged", None, None, {"reason": reason})

    def release_paper_kill_switch(self, reason: str) -> None:
        with self.tx() as conn:
            conn.execute(
                "UPDATE paper_kill_switch SET engaged = 0, engaged_at = NULL, reason = NULL "
                "WHERE id = 1"
            )
            self._audit(conn, "paper_kill_switch_released", None, None, {"reason": reason})

    def paper_kill_switch_windows(self) -> list[tuple[str, str | None]]:
        """Every `[engaged_at, released_at)` window the global paper kill switch has
        ever been on, read from `audit_log` -- there is no separate table for this,
        the switch's own engage/release events already are the history. The last
        window's end is `None` if the switch is engaged right now.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT at, event FROM audit_log WHERE event IN "
                "('paper_kill_switch_engaged', 'paper_kill_switch_released') ORDER BY id"
            ).fetchall()
        finally:
            conn.close()
        windows: list[tuple[str, str | None]] = []
        open_start: str | None = None
        for row in rows:
            if row["event"] == "paper_kill_switch_engaged":
                open_start = row["at"]
            elif row["event"] == "paper_kill_switch_released" and open_start is not None:
                windows.append((open_start, row["at"]))
                open_start = None
        if open_start is not None:
            windows.append((open_start, None))
        return windows

    # ------------------------------------------------------------ paper pause

    def is_paper_run_paused(self, run_id: int) -> bool:
        rows = self.paper_pause_rows(run_id)
        return bool(rows) and rows[-1]["kind"] == "pause"

    def pause_paper_run(self, run_id: int, reason: str = "") -> None:
        """Refuses a pause when the run is already paused: a second pause row
        back to back would be indistinguishable from a resume-then-pause, and
        the duration maths in `app.logic.stop_paper` pairs rows strictly
        pause/resume/pause/resume.
        """
        if self.is_paper_run_paused(run_id):
            raise ValueError("This run is already paused.")
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO paper_pause (run_id, kind, at, reason) VALUES (?, 'pause', ?, ?)",
                (run_id, now(), reason),
            )
            self._audit(conn, "paper_run_paused", None, run_id, {"reason": reason})

    def resume_paper_run(self, run_id: int, reason: str = "") -> None:
        if not self.is_paper_run_paused(run_id):
            raise ValueError("This run is not paused.")
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO paper_pause (run_id, kind, at, reason) VALUES (?, 'resume', ?, ?)",
                (run_id, now(), reason),
            )
            self._audit(conn, "paper_run_resumed", None, run_id, {"reason": reason})

    def paper_pause_rows(self, run_id: int) -> list[dict]:
        conn = self._connect()
        try:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM paper_pause WHERE run_id = ? ORDER BY id", (run_id,)
                )
            ]
        finally:
            conn.close()

    # --------------------------------------------------------------- paper

    def active_paper_runs(self) -> list[dict]:
        conn = self._connect()
        try:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM run WHERE mode = 'paper' AND status = 'running' "
                    "ORDER BY started_at"
                )
            ]
        finally:
            conn.close()

    def get_run(self, run_id: int) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def record_paper_event(
        self,
        run_id: int,
        kind: str,
        event_key: str,
        *,
        observed_at: str | None = None,
        symbol: str = "",
        bar_time: str | None = None,
        direction: str | None = None,
        quantity: float | None = None,
        price: float | None = None,
        detail: dict | None = None,
    ) -> bool:
        """Records an event the first time it is seen. Returns False if already recorded."""
        with self.tx() as conn:
            cur = conn.execute(
                """INSERT OR IGNORE INTO paper_event (run_id, kind, event_key, observed_at,
                   symbol, bar_time, direction, quantity, price, detail_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    kind,
                    event_key,
                    observed_at or now(),
                    symbol,
                    bar_time,
                    direction,
                    quantity,
                    price,
                    json.dumps(detail or {}, default=str),
                ),
            )
            return cur.rowcount == 1

    def paper_events(self, run_id: int) -> list[dict]:
        conn = self._connect()
        try:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM paper_event WHERE run_id = ? ORDER BY id", (run_id,)
                )
            ]
        finally:
            conn.close()

    def record_paper_decision(
        self,
        run_id: int,
        kind: str,
        event_key: str,
        *,
        observed_at: str | None = None,
        symbol: str = "",
        bar_time: str | None = None,
        reason_code: str = "",
        detail: dict | None = None,
    ) -> bool:
        """Records a decision the first time it is seen. Returns False if already recorded.

        Mirrors `record_paper_event`: idempotent on `(run_id, kind, event_key)`,
        and the table it writes to is append-only (enforced by triggers, not
        just by this method behaving itself).
        """
        with self.tx() as conn:
            cur = conn.execute(
                """INSERT OR IGNORE INTO paper_decision (run_id, observed_at, symbol,
                   bar_time, kind, reason_code, detail_json, event_key)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    observed_at or now(),
                    symbol,
                    bar_time,
                    kind,
                    reason_code,
                    json.dumps(detail or {}, default=str),
                    event_key,
                ),
            )
            return cur.rowcount == 1

    def paper_decisions(self, run_id: int) -> list[dict]:
        conn = self._connect()
        try:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM paper_decision WHERE run_id = ? ORDER BY id", (run_id,)
                )
            ]
        finally:
            conn.close()

    def set_paper_status(
        self,
        run_id: int,
        health: str,
        message: str = "",
        last_bar_time: str | None = None,
        snapshot: dict | None = None,
    ) -> None:
        with self.tx() as conn:
            conn.execute(
                """INSERT INTO paper_status (run_id, updated_at, health, message,
                   last_bar_time, snapshot_json) VALUES (?,?,?,?,?,?)
                   ON CONFLICT(run_id) DO UPDATE SET updated_at = excluded.updated_at,
                     health = excluded.health, message = excluded.message,
                     last_bar_time = COALESCE(excluded.last_bar_time, last_bar_time),
                     snapshot_json = CASE WHEN excluded.snapshot_json = '{}'
                                          THEN snapshot_json ELSE excluded.snapshot_json END""",
                (
                    run_id,
                    now(),
                    health,
                    message,
                    last_bar_time,
                    json.dumps(snapshot or {}, default=str),
                ),
            )

    def paper_status(self, run_id: int) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM paper_status WHERE run_id = ?", (run_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

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

    # --------------------------------------------------------- alert-only runs

    def is_alert_only(self, run_id: int) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT alert_only FROM paper_run_option WHERE run_id = ?", (run_id,)
            ).fetchone()
            return bool(row and row["alert_only"])
        finally:
            conn.close()

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
                for r in conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
            ]
        finally:
            conn.close()

    # ------------------------------------------------------ one-off orders

    def save_one_off_order(
        self,
        sentence: str,
        order_json: str,
        readback: str,
        created_at: str,
        *,
        slippage: dict | None = None,
    ) -> int:
        """Saves the order, and -- in the same transaction -- what slippage its
        fills should use. `slippage` is `{"kind": "pct"|"rs", "value": ...}`
        (see `nlt.costs.slippage.Slippage.to_dict`); omitted means the caller
        did not record one, which `nlt.orders.runner.saved_orders` then treats
        as the pre-2026-10-13 legacy 0.02%, not zero -- the same pattern as
        `paper_run_option` for a paper run's own options.
        """
        with self.tx() as conn:
            cur = conn.execute(
                """INSERT INTO one_off_order (sentence, order_json, readback, created_at)
                   VALUES (?,?,?,?)""",
                (sentence, order_json, readback, created_at),
            )
            order_id = cur.lastrowid
            if slippage is not None:
                conn.execute(
                    "INSERT INTO one_off_order_slippage (order_id, kind, value) VALUES (?,?,?)",
                    (order_id, slippage["kind"], slippage["value"]),
                )
            self._audit(conn, "one_off_order_saved", None, None, {"order_id": order_id})
            return order_id  # ty: ignore[invalid-return-type]

    def one_off_orders(self) -> list[dict]:
        """Every saved order, each with `slippage_kind`/`slippage_value` -- both
        None when nothing was recorded for it (a pre-2026-10-13 order).
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT o.*, s.kind AS slippage_kind, s.value AS slippage_value
                   FROM one_off_order o
                   LEFT JOIN one_off_order_slippage s ON s.order_id = o.id
                   ORDER BY o.id"""
            )
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def cancel_one_off_order(self, order_id: int) -> bool:
        """Cancels an order that has not triggered yet. Returns False if it could not be."""
        with self.tx() as conn:
            triggered = conn.execute(
                "SELECT 1 FROM one_off_event WHERE order_id = ? AND kind IN ('triggered','finished')",
                (order_id,),
            ).fetchone()
            if triggered:
                return False
            cur = conn.execute(
                "UPDATE one_off_order SET cancelled_at = ? WHERE id = ? AND cancelled_at IS NULL",
                (now(), order_id),
            )
            if cur.rowcount == 1:
                self._audit(conn, "one_off_order_cancelled", None, None, {"order_id": order_id})
            return cur.rowcount == 1

    def record_one_off_event(
        self,
        order_id: int,
        kind: str,
        event_key: str,
        *,
        observed_at: str | None = None,
        symbol: str = "",
        bar_time: str | None = None,
        side: str | None = None,
        quantity: float | None = None,
        price: float | None = None,
        detail: dict | None = None,
    ) -> bool:
        """Records an event the first time it is seen. Returns False if already recorded."""
        with self.tx() as conn:
            cur = conn.execute(
                """INSERT OR IGNORE INTO one_off_event (order_id, kind, event_key, observed_at,
                   symbol, bar_time, side, quantity, price, detail_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    order_id,
                    kind,
                    event_key,
                    observed_at or now(),
                    symbol,
                    bar_time,
                    side,
                    quantity,
                    price,
                    json.dumps(detail or {}, default=str),
                ),
            )
            return cur.rowcount == 1

    def one_off_events(self) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM one_off_event ORDER BY id").fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            d = dict(r)
            d["detail"] = json.loads(d.pop("detail_json") or "{}")
            out.append(d)
        return out

    def one_off_book(self) -> dict:
        conn = self._connect()
        try:
            row = dict(conn.execute("SELECT * FROM one_off_book WHERE id = 1").fetchone())
        finally:
            conn.close()
        row["prices"] = json.loads(row.pop("prices_json") or "{}")
        return row

    def set_one_off_book(
        self,
        *,
        health: str,
        message: str,
        cursor: str | None = None,
        prices: dict | None = None,
    ) -> None:
        with self.tx() as conn:
            conn.execute(
                """UPDATE one_off_book SET updated_at = ?, health = ?, message = ?,
                   cursor = COALESCE(?, cursor),
                   prices_json = CASE WHEN ? IS NULL THEN prices_json ELSE ? END
                   WHERE id = 1""",
                (
                    now(),
                    health,
                    message,
                    cursor,
                    None if prices is None else 1,
                    json.dumps(prices or {}, default=str),
                ),
            )

    # ------------------------------------------------------------- notifications

    def create_notification(self, source: str, source_id: int, dedupe_key: str, text: str) -> bool:
        """Writes one notification, unless `dedupe_key` has already been used.

        This is the only way a row is ever added to `notification` -- see
        `nlt.notify.notify`, which is in turn the only caller of this method.
        """
        with self.tx() as conn:
            cur = conn.execute(
                """INSERT OR IGNORE INTO notification (created_at, source, source_id,
                   dedupe_key, text) VALUES (?,?,?,?,?)""",
                (now(), source, source_id, dedupe_key, text),
            )
            return cur.rowcount == 1

    def notifications(self, limit: int = 200) -> list[dict]:
        """Newest first."""
        conn = self._connect()
        try:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM notification ORDER BY id DESC LIMIT ?", (limit,)
                )
            ]
        finally:
            conn.close()

    def unread_notification_count(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM notification WHERE read_at IS NULL"
            ).fetchone()
            return int(row["n"])
        finally:
            conn.close()

    def mark_notification_read(self, notification_id: int) -> bool:
        with self.tx() as conn:
            cur = conn.execute(
                "UPDATE notification SET read_at = ? WHERE id = ? AND read_at IS NULL",
                (now(), notification_id),
            )
            return cur.rowcount == 1

    def mark_all_notifications_read(self) -> int:
        with self.tx() as conn:
            cur = conn.execute(
                "UPDATE notification SET read_at = ? WHERE read_at IS NULL", (now(),)
            )
            return cur.rowcount
