"""
derive_api_creds.py — One-shot helper to derive Polymarket CLOB API
credentials and emit them in copy-pasteable .env format.

Why this script exists
----------------------
Polymarket's `/auth/api-key` endpoint sits behind Cloudflare and rate-
limits aggressively.  Calling it on every bot startup is fragile —
one rate-limit window means the bot can't authenticate and trades stall.

Solution: derive the creds ONCE on a clean network, paste them into
.env as CLOB_API_KEY / CLOB_API_SECRET / CLOB_API_PASSPHRASE, and let
get_clob_client() use them directly without any network call.

Usage
-----
From the repo root, with .env populated (POLYMARKET_PRIVATE_KEY,
WALLET_ADDRESS, WALLET_SIGNATURE_TYPE):

    python bot/scripts/derive_api_creds.py

Output looks like:

    SUCCESS — copy these three lines into your .env file:

    CLOB_API_KEY=...
    CLOB_API_SECRET=...
    CLOB_API_PASSPHRASE=...

If you get a 403 (Cloudflare blocked your IP):
  * wait 30-60 minutes for the rate limit to clear
  * try from a different network (mobile hotspot, VPN, your droplet)
  * the same wallet always derives the same creds — once you have them
    on any network, they'll work everywhere
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    # Load .env from the repo root
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        print("python-dotenv not installed; relying on os.environ", file=sys.stderr)

    # Make bot/ importable so we get the same config + constants the bot uses
    _SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
    _BOT_DIR     = os.path.dirname(_SCRIPTS_DIR)
    if _BOT_DIR not in sys.path:
        sys.path.insert(0, _BOT_DIR)

    private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
    wallet      = os.getenv("WALLET_ADDRESS", "").strip()
    sig_type    = int(os.getenv("WALLET_SIGNATURE_TYPE", "1"))

    if not private_key:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set in environment.", file=sys.stderr)
        return 1
    if not wallet:
        print("ERROR: WALLET_ADDRESS not set in environment.", file=sys.stderr)
        return 1

    from py_clob_client_v2 import ClobClient
    from py_clob_client_v2.constants import POLYGON

    client = ClobClient(
        host           = "https://clob.polymarket.com",
        key            = private_key,
        chain_id       = POLYGON,
        funder         = wallet,
        signature_type = sig_type,
    )

    try:
        creds = client.create_or_derive_api_key()
    except Exception as e:
        msg = str(e)
        if "403" in msg or "Cloudflare" in msg or "Sorry" in msg:
            print(
                "FAILED: Cloudflare blocked the request (HTTP 403).\n"
                "  • Wait 30-60 minutes and retry, OR\n"
                "  • Try from a different network (mobile hotspot, VPN, your droplet).\n"
                "  • Once derived from any IP, these creds work everywhere.",
                file=sys.stderr,
            )
            return 2
        print(f"FAILED: {e!r}", file=sys.stderr)
        return 3

    if not creds or not creds.api_key:
        print("FAILED: empty creds returned (no error but no values).", file=sys.stderr)
        return 4

    print("SUCCESS — copy these three lines into your .env file:")
    print()
    print(f"CLOB_API_KEY={creds.api_key}")
    print(f"CLOB_API_SECRET={creds.api_secret}")
    print(f"CLOB_API_PASSPHRASE={creds.api_passphrase}")
    print()
    print("After adding to .env, restart the bot.  get_clob_client() will use")
    print("the cached creds and no longer hit /auth/api-key on startup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
