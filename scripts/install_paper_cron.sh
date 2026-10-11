#!/usr/bin/env bash
# Installs (or re-installs) the cron line that runs paper trading every minute
# from 08:30 India time until 00:29 the next night, Monday to Friday -- MCX
# trades until 23:55. The runner itself decides what is actually due inside
# those hours.
#
# The crontab lives outside the home directory, so a rebuilt devbox loses it:
# re-run this script. Safe to run more than once.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MARK="# nlt-paper"
# The devbox clock is UTC: 03:00-18:59 UTC is 08:30-00:29 IST. (Until
# 2026-10-08 this stopped at 16:29 IST, so MCX evenings were never checked.)
LINE="* 3-18 * * 1-5 cd $ROOT && $ROOT/.venv/bin/python scripts/paper_run.py >> $ROOT/logs/paper.log 2>&1 $MARK"
mkdir -p "$ROOT/logs"
{ crontab -l 2>/dev/null | grep -v "$MARK" || true; echo "$LINE"; } | crontab -
echo "Installed:"
crontab -l | grep "$MARK"
