"""
nuclear_reset.py — One-shot "fresh start" reset.

Wraps the three existing reset operations into a single command so the
operator can wipe everything and bring up the bot under a new strategy
without leftover state from the previous run.

Steps (in order, all gated by --apply):

    PRE-FLIGHT      Snapshot current state: open positions, chain holdings,
                    DB size, log file sizes, active strategy.

    STEP 1          close_all_positions --apply --include-exiting
                    Cross-spread SELL on every open live position.  Routes
                    errors through the balance-mismatch self-heal so the DB
                    reflects chain truth even when chain is already flat.

    STEP 2          Wait `--settle` seconds (default 30) for fills to
                    confirm via WebSocket.

    STEP 3          Verify no chain holdings remain (Polymarket Data API).
                    Aborts the DB wipe if anything survives unless
                    --force-wipe is set.

    STEP 4          reset_for_new_strategy --apply [--include-paper] [--vacuum]
                    Clears positions, position_orders, processed_trade_events.

    STEP 5          Truncate logs/bot.log and logs/activity.log so the next
                    bot start has no "messages from the last run".  Skip
                    with --keep-logs.

    POST-FLIGHT     Re-snapshot state and emit a clear READY / NOT READY
                    verdict so the operator knows whether to start the bot.

Defaults to DRY-RUN.  Pass --apply to actually do the work.

Usage:
    cd bot
    python -m scripts.nuclear_reset                     # dry-run preview
    python -m scripts.nuclear_reset --apply             # commit
    python -m scripts.nuclear_reset --apply --include-paper --vacuum
    python -m scripts.nuclear_reset --apply --keep-logs
    python -m scripts.nuclear_reset --apply --skip-close   # DB wipe only
    python -m scripts.nuclear_reset --apply --force-wipe   # ignore chain remainder
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BOT_DIR)


# ---------------------------------------------------------------------------
# Visual helpers — clear, prominent banners so the operator can see exactly
# which step is running and what its purpose is.
# ---------------------------------------------------------------------------

def _banner(title: str, char: str = "=") -> None:
    bar = char * 78
    print()
    print(bar)
    print(f"  {title}")
    print(bar)


def _section(title: str) -> None:
    print()
    print(f"--- {title} ".ljust(78, "-"))


def _ok(msg: str) -> None:
    print(f"  [OK]    {msg}")


def _info(msg: str) -> None:
    print(f"  [INFO]  {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN]  {msg}")


def _err(msg: str) -> None:
    print(f"  [ERROR] {msg}")


def _explain(text: str) -> None:
    """Print a short multi-line explanation of WHAT a step does and WHY,
    so the operator never has to wonder what's about to happen."""
    print()
    for line in text.strip().splitlines():
        print(f"  > {line.strip()}")
    print()


# ---------------------------------------------------------------------------
# State snapshot helpers
# ---------------------------------------------------------------------------

def _file_size_mb(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def _file_size_kb(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / 1024


def _db_state() -> dict:
    """Snapshot DB-side state: row counts in the trading-state tables and
    breakdown of positions by status / live-vs-paper."""
    out: dict = {
        "db_path":             None,
        "db_size_mb":          0.0,
        "positions_total":     0,
        "positions_open":      0,
        "positions_exiting":   0,
        "positions_closed":    0,
        "positions_live":      0,
        "positions_paper":     0,
        "position_orders":     0,
        "trade_dedup":         0,
        "ok":                  False,
        "err":                 None,
    }
    try:
        from db import _get_conn, DB_PATH
        out["db_path"] = DB_PATH
        out["db_size_mb"] = _file_size_mb(DB_PATH)
        with _get_conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN status = 'open'    THEN 1 ELSE 0 END) AS open_,
                    SUM(CASE WHEN status = 'exiting' THEN 1 ELSE 0 END) AS exiting_,
                    SUM(CASE WHEN status = 'closed'  THEN 1 ELSE 0 END) AS closed_,
                    SUM(CASE WHEN COALESCE(is_paper, 0) = 0 THEN 1 ELSE 0 END) AS live_,
                    SUM(CASE WHEN COALESCE(is_paper, 0) = 1 THEN 1 ELSE 0 END) AS paper_
                FROM positions
            """).fetchone()
            out["positions_total"]   = int(row["total"]    or 0)
            out["positions_open"]    = int(row["open_"]    or 0)
            out["positions_exiting"] = int(row["exiting_"] or 0)
            out["positions_closed"]  = int(row["closed_"]  or 0)
            out["positions_live"]    = int(row["live_"]    or 0)
            out["positions_paper"]   = int(row["paper_"]   or 0)

            for tbl, key in (("position_orders",        "position_orders"),
                             ("processed_trade_events", "trade_dedup")):
                try:
                    out[key] = int(conn.execute(
                        f"SELECT COUNT(*) FROM {tbl}"
                    ).fetchone()[0])
                except Exception:
                    pass
        out["ok"] = True
    except Exception as e:
        out["err"] = str(e)
    return out


def _chain_state() -> dict:
    """Snapshot chain-side state via Polymarket Data API."""
    out: dict = {"queryable": False, "n_holdings": 0,
                 "holdings": [], "wallet": None, "err": None}
    try:
        from polymarket import get_data_api_positions
        from config import WALLET_ADDRESS
    except Exception as e:
        out["err"] = f"import failed: {e}"
        return out
    if not WALLET_ADDRESS:
        out["err"] = "WALLET_ADDRESS not configured in .env"
        return out
    out["wallet"] = WALLET_ADDRESS
    try:
        positions = get_data_api_positions(WALLET_ADDRESS) or []
    except Exception as e:
        out["err"] = f"Data API call failed: {e}"
        return out
    nonzero = [p for p in positions if float(p.get("size", 0) or 0) > 0]
    out["queryable"]  = True
    out["n_holdings"] = len(nonzero)
    out["holdings"]   = nonzero
    return out


def _log_state() -> dict:
    out = {"bot_log_kb":      0.0,
           "activity_log_kb": 0.0}
    logs_dir = os.path.join(_BOT_DIR, "logs")
    out["bot_log_kb"]      = _file_size_kb(os.path.join(logs_dir, "bot.log"))
    out["activity_log_kb"] = _file_size_kb(os.path.join(logs_dir, "activity.log"))
    return out


def _active_strategy() -> str:
    try:
        from config import ACTIVE_STRATEGY
        return ACTIVE_STRATEGY or "(unset)"
    except Exception as e:
        return f"(could not load: {e})"


def _print_state(label: str, db: dict, chain: dict, logs: dict) -> None:
    """Render a state snapshot block — used both pre-flight and post-flight
    so the operator can compare before/after at a glance."""
    _section(f"STATE SNAPSHOT — {label}")

    # DB ----------------------------------------------------------------------
    if db["ok"]:
        print(f"  DB:        {db['db_path']}")
        print(f"             size           : {db['db_size_mb']:.2f} MB")
        print(f"             positions      : {db['positions_total']:>4} total  "
              f"({db['positions_live']} live / {db['positions_paper']} paper)")
        print(f"               open         : {db['positions_open']:>4}")
        print(f"               exiting      : {db['positions_exiting']:>4}")
        print(f"               closed       : {db['positions_closed']:>4}")
        print(f"             position_orders: {db['position_orders']:>4} rows")
        print(f"             trade dedup    : {db['trade_dedup']:>4} rows")
    else:
        print(f"  DB:        ERROR -- {db['err']}")

    # Chain -------------------------------------------------------------------
    if chain["queryable"]:
        wallet = chain["wallet"] or "?"
        print(f"  CHAIN:     wallet         : {wallet[:10]}...{wallet[-6:]}")
        print(f"             open holdings  : {chain['n_holdings']}")
        if chain["n_holdings"] > 0:
            for p in chain["holdings"][:10]:
                title = (p.get("title", "") or "")[:50]
                size  = float(p.get("size", 0) or 0)
                avg   = float(p.get("avg_price", 0) or 0)
                print(f"               - {title:<50} size={size:>9.4f} avg={avg:.4f}")
            if chain["n_holdings"] > 10:
                print(f"               ... and {chain['n_holdings'] - 10} more")
    else:
        print(f"  CHAIN:     UNAVAILABLE -- {chain['err']}")

    # Logs --------------------------------------------------------------------
    print(f"  LOGS:      bot.log        : {logs['bot_log_kb']:>7.1f} KB")
    print(f"             activity.log   : {logs['activity_log_kb']:>7.1f} KB")


# ---------------------------------------------------------------------------
# Sub-script runner
# ---------------------------------------------------------------------------

def _run_subscript(name: str, extra_args: list[str]) -> int:
    """Run one of the existing scripts as a child process so its argparse
    handling and side-effects stay isolated."""
    cmd = [sys.executable, "-m", f"scripts.{name}", *extra_args]
    print(f"  $ {' '.join(cmd)}")
    print()
    res = subprocess.run(cmd, cwd=_BOT_DIR)
    return res.returncode


def _truncate_logs() -> list[str]:
    """Truncate bot.log and activity.log in-place so the next bot start
    sees an empty file (preserving the inode for any tail -f sessions)."""
    truncated: list[str] = []
    logs_dir = os.path.join(_BOT_DIR, "logs")
    for name in ("bot.log", "activity.log"):
        path = os.path.join(logs_dir, name)
        if os.path.exists(path):
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.truncate(0)
                truncated.append(name)
            except Exception as e:
                _warn(f"could not truncate {name}: {e}")
    return truncated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Actually run the reset.  Without this, dry-run.")
    ap.add_argument("--skip-close", action="store_true",
                    help="Skip step 1 (close_all_positions).  DB wipe + log "
                         "truncate only.  Use when chain is already flat.")
    ap.add_argument("--settle", type=int, default=30,
                    help="Seconds to wait between close-all and DB wipe so "
                         "fills can confirm via WebSocket (default: 30)")
    ap.add_argument("--include-paper", action="store_true",
                    help="Also clear paper positions in step 4")
    ap.add_argument("--vacuum", action="store_true",
                    help="Run VACUUM after the DB wipe to reclaim disk space")
    ap.add_argument("--keep-logs", action="store_true",
                    help="Skip step 5 (truncate logs)")
    ap.add_argument("--force-wipe", action="store_true",
                    help="Wipe the DB in step 4 even if chain holdings remain. "
                         "DANGEROUS -- you'll have orphan chain holdings the "
                         "bot won't know about.  Only use when you'll close "
                         "those manually via the Polymarket UI.")
    args = ap.parse_args()

    mode = "APPLY" if args.apply else "DRY RUN"
    started = datetime.now().isoformat(timespec="seconds")

    _banner(f"NUCLEAR RESET  --  mode: {mode}")
    print(f"  Purpose:        Wipe ALL trading state so the bot can be re-launched")
    print(f"                  under a new strategy with zero leftover artifacts.")
    print(f"  Started:        {started}")
    print(f"  bot dir:        {_BOT_DIR}")
    print(f"  active strategy:{_active_strategy()}")
    print(f"  options:        skip-close={args.skip_close}   "
          f"settle={args.settle}s   include-paper={args.include_paper}   "
          f"vacuum={args.vacuum}")
    print(f"                  keep-logs={args.keep_logs}   "
          f"force-wipe={args.force_wipe}")

    # ---- PRE-FLIGHT: snapshot current state ----------------------------
    _banner("PRE-FLIGHT — current state of the bot")
    _explain("""
        This is what the bot looks like RIGHT NOW.  Compare the numbers
        below to the POST-FLIGHT snapshot at the end to confirm the
        reset did its job (open=0, exiting=0, holdings=0, log files
        ~0 KB).
    """)
    db_pre    = _db_state()
    chain_pre = _chain_state()
    logs_pre  = _log_state()
    _print_state("BEFORE", db_pre, chain_pre, logs_pre)

    # In dry-run mode, just describe the plan and exit.
    if not args.apply:
        _banner("DRY RUN — planned actions (not executed)")
        plan = []
        if args.skip_close:
            plan.append("STEP 1  -- SKIPPED (--skip-close)")
            plan.append("STEP 2  -- SKIPPED (--skip-close)")
        else:
            plan.append(f"STEP 1  close_all_positions --apply --include-exiting")
            plan.append(f"          would attempt to sell "
                        f"{db_pre['positions_open'] + db_pre['positions_exiting']} "
                        f"open/exiting position(s)")
            plan.append(f"STEP 2  wait {args.settle}s for fill confirmations")
        plan.append(f"STEP 3  verify chain holdings == 0   "
                    + (f"(chain currently shows {chain_pre['n_holdings']})"
                       if chain_pre['queryable'] else "(chain unavailable)"))
        cmd4 = "reset_for_new_strategy --apply"
        if args.include_paper: cmd4 += " --include-paper"
        if args.vacuum:        cmd4 += " --vacuum"
        plan.append(f"STEP 4  {cmd4}")
        plan.append(f"          would clear "
                    f"{db_pre['positions_total']} positions + "
                    f"{db_pre['position_orders']} ledger rows + "
                    f"{db_pre['trade_dedup']} dedup rows")
        if args.keep_logs:
            plan.append("STEP 5  -- SKIPPED (--keep-logs)")
        else:
            plan.append(f"STEP 5  truncate bot.log ({logs_pre['bot_log_kb']:.1f} KB) "
                        f"+ activity.log ({logs_pre['activity_log_kb']:.1f} KB)")
        for line in plan:
            print(f"  {line}")
        print()
        _info("Re-run with --apply to commit.")
        return 0

    # ---- STEP 1: close_all_positions ------------------------------------
    if args.skip_close:
        _banner("STEP 1  close_all_positions  --  SKIPPED (--skip-close)")
        _info("Trusting that the chain is already flat.  Step 3 will verify.")
    else:
        _banner("STEP 1  close_all_positions --apply --include-exiting")
        _explain("""
            What this does:
              For every position currently in 'open' or 'exiting' status,
              place a marketable cross-spread SELL.  If the order is
              rejected because chain shares are already 0 (e.g. you closed
              them manually via the Polymarket UI), the EXIT BALANCE-
              MISMATCH self-heal path will silently mark the DB row closed
              with the approximate realized PnL.

            What you'll see below:
              - One line per position with its target sell price
              - For positions still on chain: "SELL placed @ X (will fill via WS)"
              - For positions already off chain: "closed via self-heal"
              - A final summary block from close_all_positions itself
        """)
        rc = _run_subscript("close_all_positions",
                            ["--apply", "--include-exiting"])
        if rc != 0:
            _warn(f"close_all_positions exited with code {rc} — continuing")
        else:
            _ok("close_all_positions completed")

        # ---- STEP 2: wait for fills ------------------------------------
        _banner(f"STEP 2  wait {args.settle}s for fill confirmations")
        _explain(f"""
            Why we wait:
              When a real cross-spread SELL goes on the book, the WebSocket
              fill handler needs a few seconds to receive the on-chain
              CONFIRMED event and flip the DB row to status='closed'.
              Skipping this wait risks step 3 reporting "chain not flat"
              for orders that are about to settle.

            Self-healed positions (chain already at 0) do NOT need this
            wait — they were closed synchronously above — but we wait the
            full {args.settle}s to be safe across mixed cases.
        """)
        for i in range(args.settle, 0, -5):
            print(f"  ...{i:>3}s remaining")
            time.sleep(min(5, i))
        _ok("settle window complete")

    # ---- STEP 3: verify chain is flat -----------------------------------
    _banner("STEP 3  verify chain holdings == 0")
    _explain("""
        What this does:
          Fetch the wallet's open token positions from the Polymarket
          Data API (the on-chain source of truth) and check the count.

        Why it matters:
          If holdings remain on chain after step 1, those tokens become
          ORPHANS the bot can't see — they sit in your wallet generating
          PnL with no exit ladder, no stop loss, no risk awareness.  We
          refuse to wipe the DB unless either chain is flat OR the
          operator explicitly passes --force-wipe (in which case they're
          accepting responsibility to close those holdings manually).
    """)
    chain_now = _chain_state()
    if not chain_now["queryable"]:
        _warn(f"chain unavailable -- {chain_now['err']}")
        _warn("proceeding without verification (treat post-flight numbers as authoritative)")
    else:
        n = chain_now["n_holdings"]
        if n == 0:
            _ok(f"chain confirmed FLAT (0 open holdings)")
        elif args.force_wipe:
            _warn(f"chain still has {n} open holding(s) BUT --force-wipe is set")
            for p in chain_now["holdings"][:5]:
                title = (p.get("title", "") or "")[:60]
                size  = float(p.get("size", 0) or 0)
                _warn(f"  orphan-to-be: {title}  size={size:.4f}")
            _warn("DB wipe will proceed; you must close these manually on Polymarket UI")
        else:
            _err(f"chain still has {n} open holding(s) — ABORTING DB wipe")
            for p in chain_now["holdings"][:10]:
                title = (p.get("title", "") or "")[:60]
                size  = float(p.get("size", 0) or 0)
                avg   = float(p.get("avg_price", 0) or 0)
                _err(f"  {title}  size={size:.4f}  avg={avg:.4f}")
            print()
            _info("To resolve, choose one of:")
            _info("  a) wait longer + re-run (fills may still be confirming)")
            _info("  b) close them manually via the Polymarket web UI")
            _info("  c) re-run with --force-wipe (orphans will exist on chain "
                  "but be invisible to the bot)")
            return 2

    # ---- STEP 4: DB reset -----------------------------------------------
    _banner("STEP 4  reset_for_new_strategy --apply")
    _explain("""
        What this does:
          Truncates three trading-state tables in the SQLite DB:
            - positions               (all rows: open, exiting, closed)
            - position_orders         (per-CLOB-order ledger)
            - processed_trade_events  (WS fill dedup state)

        What this PRESERVES:
          - activity_log              (audit trail of every bot action)
          - monitor_health            (per-cycle health snapshots)
          - temp_events / temp_outcomes  (Polymarket discovery scan history)
          - event_resolutions         (past winners — used by analytics)
          - all weather pipeline tables (forecasts, observations)
          - all ML / climatology tables

        The reset is itself logged to activity_log under category='RESET'
        so the wipe is forever auditable in the dashboard.
    """)
    reset_args = ["--apply"]
    if args.include_paper:
        reset_args.append("--include-paper")
    if args.vacuum:
        reset_args.append("--vacuum")
    rc = _run_subscript("reset_for_new_strategy", reset_args)
    if rc != 0:
        _err(f"reset_for_new_strategy exited with code {rc}")
        _err("ABORTING — DB state is uncertain; investigate before restarting bot")
        return 3
    _ok("reset_for_new_strategy completed")

    # ---- STEP 5: truncate logs ------------------------------------------
    if args.keep_logs:
        _banner("STEP 5  truncate logs  --  SKIPPED (--keep-logs)")
        _info("bot.log and activity.log retained at their current size.")
    else:
        _banner("STEP 5  truncate bot.log + activity.log")
        _explain("""
            What this does:
              Truncates each log file in place (preserves the inode so any
              tail -f or dashboard tail session keeps working).  After
              this, the next bot start writes its first line into a 0-byte
              file — so when you scroll back, you see ONLY the new run.
        """)
        truncated = _truncate_logs()
        if truncated:
            for name in truncated:
                _ok(f"truncated logs/{name}")
        else:
            _info("no log files found to truncate (skipped)")

    # ---- POST-FLIGHT: snapshot final state + verdict --------------------
    _banner("POST-FLIGHT — final state of the bot")
    _explain("""
        This is what the bot looks like AFTER the reset.  Compare to the
        PRE-FLIGHT snapshot above; everything trading-related should now
        read 0.  Historical/audit tables are intentionally preserved.
    """)
    db_post    = _db_state()
    chain_post = _chain_state()
    logs_post  = _log_state()
    _print_state("AFTER", db_post, chain_post, logs_post)

    # Verdict — be opinionated.  Tell the operator green-or-red whether
    # they're safe to start the bot.
    _banner("VERDICT — is the bot ready for a fresh start?")
    issues: list[str] = []
    if db_post["positions_open"] > 0:
        issues.append(
            f"DB still shows {db_post['positions_open']} 'open' position(s) "
            f"— reset_for_new_strategy did not fully clear"
        )
    if db_post["positions_exiting"] > 0:
        issues.append(
            f"DB still shows {db_post['positions_exiting']} 'exiting' position(s)"
        )
    if db_post["position_orders"] > 0:
        issues.append(
            f"DB still shows {db_post['position_orders']} position_orders rows"
        )
    if db_post["trade_dedup"] > 0:
        issues.append(
            f"DB still shows {db_post['trade_dedup']} processed_trade_events rows"
        )
    if chain_post["queryable"] and chain_post["n_holdings"] > 0 and not args.force_wipe:
        issues.append(
            f"chain still has {chain_post['n_holdings']} open holding(s)"
        )

    finished = datetime.now().isoformat(timespec="seconds")
    print(f"  Started:    {started}")
    print(f"  Finished:   {finished}")
    print()

    if not issues:
        print("  >>>>>  STATUS: READY FOR FRESH START  <<<<<")
        print()
        _ok("All trading-state tables are empty")
        if chain_post["queryable"]:
            _ok(f"Chain is flat (0 open holdings on wallet "
                f"{(chain_post['wallet'] or '')[:10]}...)")
        if not args.keep_logs:
            _ok(f"Log files truncated "
                f"(bot.log={logs_post['bot_log_kb']:.1f}KB, "
                f"activity.log={logs_post['activity_log_kb']:.1f}KB)")
        if args.include_paper:
            _ok("Paper positions also cleared (--include-paper)")
        if args.vacuum:
            _ok(f"DB vacuumed (now {db_post['db_size_mb']:.2f} MB)")
        print()
        print("  Next steps:")
        print("    1. Confirm .env has the strategy you want:")
        print(f"         ACTIVE_STRATEGY currently = {_active_strategy()}")
        print("       (top_k_hedged / market_price_value / top_bin_value)")
        print("    2. Start the bot:")
        print("         cd bot && python main.py")
        print("    3. Watch the first cycle for funnel telemetry, risk-cap")
        print("       warnings, and the first signal generation.")
        return 0
    else:
        print("  >>>>>  STATUS: NOT READY  <<<<<")
        print()
        for issue in issues:
            _err(issue)
        print()
        _info("Investigate the issues above before starting the bot.  "
              "Re-running this script may or may not resolve them depending "
              "on the cause.")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
