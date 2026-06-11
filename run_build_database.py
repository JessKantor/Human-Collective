#!/usr/bin/env python3
"""
Auto-restart wrapper for build_database.py.

If Python quits or crashes mid-build, this script waits a few seconds and
launches it again in --resume mode so progress is not lost.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = SCRIPT_DIR / "build_database.py"
COMPLETE_MARKER = SCRIPT_DIR / "build_complete.json"
RESTART_DELAY_SECONDS = 10
MAX_RESTARTS = 200


def main() -> int:
    extra_args = sys.argv[1:]
    restarts = 0

    if COMPLETE_MARKER.exists() and "--fresh" not in extra_args:
        print("Build already complete (build_complete.json exists).")
        print("Run with --fresh to start over:")
        print("  python3 run_build_database.py --fresh")
        return 0

    print("Watchdog started — will auto-restart build_database.py on exit.")
    print("Press Ctrl+C to stop the watchdog.\n")

    while restarts < MAX_RESTARTS:
        cmd = [sys.executable, str(BUILD_SCRIPT)]
        if "--fresh" not in extra_args:
            cmd.append("--resume")
        cmd.extend(extra_args)

        print("=" * 60)
        print(f"Launching: {' '.join(cmd)}")
        print("=" * 60 + "\n")

        result = subprocess.run(cmd, cwd=SCRIPT_DIR)

        if COMPLETE_MARKER.exists() and result.returncode == 0:
            print("\nBuild finished successfully.")
            return 0

        restarts += 1
        print(
            f"\nbuild_database.py exited with code {result.returncode}. "
            f"Restarting in {RESTART_DELAY_SECONDS}s "
            f"(restart {restarts}/{MAX_RESTARTS})..."
        )
        print("Press Ctrl+C now to stop the watchdog.")

        try:
            time.sleep(RESTART_DELAY_SECONDS)
        except KeyboardInterrupt:
            print("\nWatchdog stopped. Run again to resume:")
            print("  python3 run_build_database.py")
            return 1

    print(f"Reached max restarts ({MAX_RESTARTS}). Stopping.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
