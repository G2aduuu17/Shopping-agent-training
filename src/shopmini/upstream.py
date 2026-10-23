"""Read-only boundary to a pinned, separate upstream checkout."""
import os
import subprocess
import sys
from pathlib import Path

UPSTREAM_COMMIT = "dc6c9af8e48c8e6325101b39d1d20060f69e4221"
UPSTREAM_URL = "https://github.com/YYHDBL/shopping-grpo-longhorizon.git"
PRODUCT_SHA256 = "57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f"
MINI_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = Path(os.environ.get("SHOPMINI_UPSTREAM_DIR", MINI_ROOT / "upstream")).resolve()


def require_upstream():
    if not (UPSTREAM_ROOT / ".git").exists():
        raise RuntimeError(f"Missing upstream checkout: {UPSTREAM_ROOT}")
    # Per-command trust for our explicitly selected checkout; no global Git config changes.
    result = subprocess.run(["git", "-c", f"safe.directory={UPSTREAM_ROOT.as_posix()}",
                             "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip() != UPSTREAM_COMMIT:
        raise RuntimeError(f"Upstream must be pinned to {UPSTREAM_COMMIT}")
    subprocess.run(["git", "-c", f"safe.directory={UPSTREAM_ROOT.as_posix()}",
                    "-C", str(UPSTREAM_ROOT), "diff", "--exit-code", "HEAD", "--"],
                   check=True, stdout=subprocess.DEVNULL)
    return UPSTREAM_ROOT


def enable_upstream():
    root = require_upstream()
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root
