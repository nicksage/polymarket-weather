#!/usr/bin/env python3
"""
install_hooks.py — Copy versioned git-hook templates into .git/hooks/.

Git won't run hooks unless they live under .git/hooks/, but that
directory isn't tracked by git itself.  We keep the source-of-truth
hooks under bot/scripts/git_hooks/ (versioned, code-reviewed) and copy
them into place with this script.

Run once per fresh clone, on every machine that will commit:
    python bot/scripts/install_hooks.py

Safe to re-run — idempotent overwrite of existing hooks.  Cross-
platform: works on Linux, macOS, Windows (Git Bash sets the executable
bit appropriately).
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_SOURCE_DIR = REPO_ROOT / "bot" / "scripts" / "git_hooks"
HOOK_DEST_DIR   = REPO_ROOT / ".git" / "hooks"


def main() -> int:
    if not HOOK_SOURCE_DIR.is_dir():
        print(f"FATAL: hook source dir not found: {HOOK_SOURCE_DIR}",
              file=sys.stderr)
        return 1
    if not HOOK_DEST_DIR.is_dir():
        print(f"FATAL: {HOOK_DEST_DIR} not found — is this a git repo?",
              file=sys.stderr)
        return 1

    installed = []
    for src in HOOK_SOURCE_DIR.iterdir():
        if not src.is_file() or src.name.startswith("."):
            continue
        dst = HOOK_DEST_DIR / src.name
        shutil.copy2(src, dst)
        # Set executable bits (POSIX).  On Windows this is a no-op
        # but doesn't hurt — Git on Windows respects the shebang.
        try:
            mode = os.stat(dst).st_mode
            os.chmod(dst, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            pass
        installed.append(dst)

    if not installed:
        print(f"No hooks found in {HOOK_SOURCE_DIR}")
        return 1

    print(f"Installed {len(installed)} hook(s):")
    for d in installed:
        print(f"  {d.relative_to(REPO_ROOT)}")
    print()
    print("Verify with a dry run:")
    print(f"  python {Path('bot/scripts/deploy_safety_check.py')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())