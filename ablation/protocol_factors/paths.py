"""Package-local directories."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
