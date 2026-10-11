"""One-off conditional orders: "when this happens, do that, once" -- paper only.

The strategy recipe in `nlt.spec` describes something that trades again and
again and is judged by a backtest. These are different: Zerodha's Alert Trigger
Orders (ATO) are the model -- a condition is watched, and when it is met an
order goes through once and the order is finished.

Justin's decisions (2026-10-08), each explained to him as a trade-off:

  * **Paper only.** Nothing here can reach Zerodha. Real ATOs would mean letting
    the app place orders, which has not been decided.
  * **A missing stop loss is allowed, but said out loud.** One-off orders do not
    need one; the readback warns and the screen offers to add one.
  * **A P&L exit looks only at this app's own paper positions,** never at the
    real Zerodha account.
"""
