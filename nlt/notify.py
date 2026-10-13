"""The one delivery point for every notification this app ever sends.

Today that means a single row in `notification`, which the header bell's
unread count and the Alerts page both read. If email is ever bolted on, it is
bolted on here, inside this one function, so every caller -- the one-off order
checker, the alert-only strategy runner -- stays exactly the same.

Idempotent by design: `dedupe_key` is UNIQUE in the table, so calling this
twice with the same key writes the row once. Callers build a key that can
only ever mean "this exact thing, once" -- an order id for a one-off alert
that can only fire once, or a strategy's decision key for an alert-only run.
"""

from __future__ import annotations

from nlt.store.db import Store


def notify(store: Store, source: str, source_id: int, dedupe_key: str, text: str) -> bool:
    """Deliver one notification. Returns True if this call created it, False
    if a notification with this `dedupe_key` already existed.
    """
    return store.create_notification(source, source_id, dedupe_key, text)
