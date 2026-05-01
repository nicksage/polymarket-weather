"""
wallet.py — Wallet balance sync layer.

Provides `get_effective_bankroll(client)` which returns the smaller of:
  * INITIAL_BANKROLL                        (your intentional tradeable cap)
  * actual wallet USDC balance - reserve (your physical cap)

This protects against the "I withdrew USDC from the proxy wallet but forgot
to update INITIAL_BANKROLL" failure mode — without it, the bot would size
against the stale config and hit on-chain rejections at order time.

The shortfall is silent (no abort): the bot just sizes against the lower
cap and logs an INFO line so you can see in the logs that the cap was
hit.  In paper mode, or when WALLET_BALANCE_CHECK_ENABLED=False, this
module is a passthrough that returns INITIAL_BANKROLL unchanged.

Caching: balance is cached per-process for WALLET_BALANCE_REFRESH_MIN
minutes.  Within a single trading cycle, all sizing calls see the same
balance reading.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from config import (
    INITIAL_BANKROLL,
    PAPER_TRADE,
    WALLET_BALANCE_CHECK_ENABLED,
    WALLET_BALANCE_RESERVE_USDC,
    WALLET_BALANCE_REFRESH_MIN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache (module-level — persists for process lifetime)
# ---------------------------------------------------------------------------

_cache_balance: float | None = None
_cache_at_epoch: float = 0.0
_last_logged_scale_down: float | None = None   # so we don't spam INFO every cycle


def _cache_is_fresh() -> bool:
    """True iff we have a cached reading less than WALLET_BALANCE_REFRESH_MIN
    minutes old."""
    if _cache_balance is None:
        return False
    age_sec = time.time() - _cache_at_epoch
    return age_sec < WALLET_BALANCE_REFRESH_MIN * 60.0


def clear_cache() -> None:
    """Test hook + manual override — drops the cached balance so the next
    call goes back to the API."""
    global _cache_balance, _cache_at_epoch, _last_logged_scale_down
    _cache_balance = None
    _cache_at_epoch = 0.0
    _last_logged_scale_down = None


# ---------------------------------------------------------------------------
# Raw balance fetch via CLOB client
# ---------------------------------------------------------------------------

def get_wallet_usdc_balance(client: Any | None) -> float | None:
    """Return the wallet's free USDC balance via the CLOB client.

    Uses py_clob_client_v2's get_balance_allowance(COLLATERAL) which
    returns the USDC balance in micro-units (1e6 = 1 USDC).  The CLOB
    client must be authenticated (key + API creds set) AND constructed
    with the right `funder=WALLET_ADDRESS` and `signature_type=2`
    (POLY_GNOSIS_SAFE for MetaMask-connected wallets) — this is what
    execution.get_clob_client() does in non-paper mode.

    Returns None on:
      * paper mode (no client)
      * client is None for any reason
      * CLOB call raises (logged as warning)
      * response shape is unexpected
    """
    if client is None:
        return None

    try:
        # v2 import path — same class names + method, just a different
        # package.  Migrated 2026-04-29 alongside execution.py for the
        # order_version_mismatch fix.
        from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        result = client.get_balance_allowance(params)
        # Result shape: {"balance": "12345678", "allowance": "..."} where
        # balance is a stringified integer in USDC micro-units (6 decimals)
        raw = result.get("balance") if isinstance(result, dict) else None
        if raw is None:
            logger.warning(f"wallet: unexpected balance response shape: {result}")
            return None
        return float(raw) / 1_000_000.0
    except Exception as e:
        logger.warning(f"wallet: USDC balance fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API — effective bankroll
# ---------------------------------------------------------------------------

def get_effective_bankroll(client: Any | None) -> float:
    """Return the bankroll the bot should actually size against.

    Logic:
      * Paper mode or check disabled → INITIAL_BANKROLL unchanged
      * Live mode + balance fetch succeeds:
          effective = min(INITIAL_BANKROLL, max(0, balance - reserve))
          Logs INFO when the wallet is the binding cap (rate-limited so
          we don't spam every cycle when the situation is stable).
      * Live mode + balance fetch fails:
          Falls back to INITIAL_BANKROLL.  Logs warning.  This is defensive —
          we'd rather size against the stale config than refuse to trade.
    """
    if PAPER_TRADE or not WALLET_BALANCE_CHECK_ENABLED or client is None:
        return float(INITIAL_BANKROLL)

    global _cache_balance, _cache_at_epoch, _last_logged_scale_down

    if not _cache_is_fresh():
        balance = get_wallet_usdc_balance(client)
        if balance is None:
            # Fetch failed — fall back to config but don't poison the cache
            logger.warning(
                f"wallet: using INITIAL_BANKROLL={INITIAL_BANKROLL:.2f} "
                f"as effective bankroll (balance check failed)"
            )
            return float(INITIAL_BANKROLL)
        _cache_balance = balance
        _cache_at_epoch = time.time()

    balance = _cache_balance or 0.0
    available = max(0.0, balance - WALLET_BALANCE_RESERVE_USDC)
    effective = min(float(INITIAL_BANKROLL), available)

    # Log the scale-down event, but rate-limited: only log when the bound
    # CHANGES (config-bound vs wallet-bound) or hasn't been logged in 1h.
    bound_by_wallet = available < float(INITIAL_BANKROLL)
    if bound_by_wallet:
        now = time.time()
        if (_last_logged_scale_down is None
                or now - _last_logged_scale_down > 3600):
            try:
                from activity import log_activity
                log_activity(
                    "RISK", level="WARN",
                    message=(
                        f"bankroll capped by wallet: effective=${effective:.2f} "
                        f"(wallet=${balance:.2f} - reserve "
                        f"${WALLET_BALANCE_RESERVE_USDC:.2f} < "
                        f"INITIAL_BANKROLL=${float(INITIAL_BANKROLL):.2f})"
                    ),
                    wallet_balance=balance, effective_bankroll=effective,
                    initial_bankroll=float(INITIAL_BANKROLL),
                    reserve=WALLET_BALANCE_RESERVE_USDC,
                )
            except Exception:
                logger.info(
                    f"wallet: effective bankroll capped at ${effective:.2f} "
                    f"(wallet ${balance:.2f} - reserve "
                    f"${WALLET_BALANCE_RESERVE_USDC:.2f})"
                )
            _last_logged_scale_down = now

    return effective
