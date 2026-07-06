"""Central env loader. Import DB_PATH, LOG_DIR, API keys from here."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")

DB_PATH = os.getenv("DB_PATH", str(REPO_ROOT / "db" / "main.db"))
LOG_DIR = os.getenv("LOG_DIR", str(REPO_ROOT / "logs"))
os.makedirs(LOG_DIR, exist_ok=True)

# Weather API keys (loaded but not validated here)
NOAA_API_TOKEN         = os.getenv("NOAA_API_TOKEN")
TOMORROWIO_API_KEY     = os.getenv("TOMORROWIO_API_KEY")
VISUAL_CROSSING_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
TWC_API_KEY            = os.getenv("TWC_API_KEY")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
