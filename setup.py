"""
setup.py — One-time setup script for the Polymarket Weather Trading Bot.

Run this once after cloning the repository to install all dependencies,
initialize the databases, verify API connectivity, and create the .env
file if it doesn't exist.

    python setup.py

What it does:
    1. Installs Python dependencies from requirements.txt
    2. Creates .env from .env.example if .env doesn't exist
    3. Creates the bot/data/ and bot/logs/ directories
    4. Initializes the live database (signals.db) with all tables
    5. Initializes the backtest database (backtest.db)
    6. Runs a basic health check (Python version, key imports)
    7. Prints next steps

What it does NOT do:
    - Fill in your API keys (you must edit .env manually)
    - Start the bot
    - Place any trades
"""

import os
import subprocess
import sys
import shutil


def _print_header(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def _print_ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _print_warn(msg: str) -> None:
    print(f"  [!!] {msg}")


def _print_fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    bot_dir = os.path.join(root, "bot")

    _print_header("Polymarket Weather Bot Setup")
    print(f"  Project root: {root}")
    print(f"  Python: {sys.version.split()[0]}")

    # ------------------------------------------------------------------
    # Step 1: Check Python version
    # ------------------------------------------------------------------
    _print_header("Step 1: Python version check")
    if sys.version_info < (3, 11):
        _print_fail(f"Python 3.11+ required (you have {sys.version_info.major}.{sys.version_info.minor})")
        return 1
    _print_ok(f"Python {sys.version_info.major}.{sys.version_info.minor}")

    # ------------------------------------------------------------------
    # Step 2: Install dependencies
    # ------------------------------------------------------------------
    _print_header("Step 2: Installing dependencies")
    req_file = os.path.join(root, "requirements.txt")
    if not os.path.exists(req_file):
        _print_fail("requirements.txt not found")
        return 1

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_file],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        _print_fail("pip install failed:")
        print(result.stderr[-500:] if result.stderr else result.stdout[-500:])
        return 1
    _print_ok("All dependencies installed")

    # ------------------------------------------------------------------
    # Step 3: Create .env if missing
    # ------------------------------------------------------------------
    _print_header("Step 3: Environment configuration")
    env_file = os.path.join(root, ".env")
    env_example = os.path.join(root, ".env.example")

    if os.path.exists(env_file):
        _print_ok(".env already exists")
    elif os.path.exists(env_example):
        shutil.copy2(env_example, env_file)
        _print_ok("Created .env from .env.example")
        _print_warn("You MUST edit .env and fill in your API keys before running the bot")
    else:
        _print_warn(".env and .env.example not found - you will need to create .env manually")
        _print_warn("See README.md for required environment variables")

    # ------------------------------------------------------------------
    # Step 4: Create directories
    # ------------------------------------------------------------------
    _print_header("Step 4: Creating directories")
    for subdir in ["data", "logs"]:
        path = os.path.join(bot_dir, subdir)
        os.makedirs(path, exist_ok=True)
        _print_ok(f"bot/{subdir}/")

    # ------------------------------------------------------------------
    # Step 5: Initialize databases
    # ------------------------------------------------------------------
    _print_header("Step 5: Initializing databases")

    # Need to be in bot/ for imports to work
    original_dir = os.getcwd()
    os.chdir(bot_dir)
    sys.path.insert(0, bot_dir)

    try:
        from db import init_db
        init_db()
        _print_ok("signals.db initialized")
    except Exception as e:
        _print_fail(f"signals.db initialization failed: {e}")

    try:
        from backtest.backtest_db import init_backtest_db
        init_backtest_db()
        _print_ok("backtest.db initialized")
    except Exception as e:
        _print_fail(f"backtest.db initialization failed: {e}")

    os.chdir(original_dir)

    # ------------------------------------------------------------------
    # Step 6: Verify key imports
    # ------------------------------------------------------------------
    _print_header("Step 6: Import verification")

    checks = [
        ("httpx", "HTTP client"),
        ("tenacity", "Retry logic"),
        ("apscheduler", "Scheduler"),
        ("scipy", "Probability math"),
        ("numpy", "Numerical computing"),
        ("astral", "Sunrise/sunset"),
        ("timezonefinder", "Timezone resolution"),
        ("streamlit", "Dashboard"),
        ("plotly", "Charts"),
        ("pandas", "Data frames"),
        ("dotenv", "Config loading"),
    ]

    all_ok = True
    for module, desc in checks:
        try:
            __import__(module)
            _print_ok(f"{module} ({desc})")
        except ImportError:
            _print_fail(f"{module} ({desc}) - not installed")
            all_ok = False

    # Optional: py_clob_client (only needed for live trading)
    try:
        __import__("py_clob_client")
        _print_ok("py_clob_client (Polymarket CLOB - live trading)")
    except ImportError:
        _print_warn("py_clob_client not installed (only needed for live trading, not paper mode)")

    # ------------------------------------------------------------------
    # Step 7: Summary
    # ------------------------------------------------------------------
    _print_header("Setup complete")

    if not all_ok:
        print("\n  Some dependencies failed to install. Run:")
        print("    pip install -r requirements.txt")
        print("  and try again.\n")
        return 1

    print("""
  Next steps:

    1. Edit .env and fill in your API keys:
       - VISUAL_CROSSING_API_KEY (required)
       - POLYMARKET_PRIVATE_KEY (required for live trading)
       - TOMORROWIO_API_KEY (optional)

    2. Review .env settings:
       - PAPER_TRADE=true (default, recommended for first run)
       - ACTIVE_STRATEGY=top_bin_value (or edge_disagreement)
       - BANKROLL_USDC=1000 (set your paper bankroll)

    3. Start the bot:
       cd bot
       python main.py

    4. Start the dashboard (in a separate terminal):
       cd bot
       streamlit run dashboard.py

  For more details, see README.md.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
