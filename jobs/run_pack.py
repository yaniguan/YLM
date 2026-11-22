"""Generic scheduler-agnostic launcher for training pack files under jobs/packs/*.txt.

Runs up to --parallel commands concurrently on one GPU or workstation without assuming
any specific cluster scheduler, account, or filesystem layout.

  python jobs/run_pack.py --pack jobs/packs/v1_0.txt --parallel 2
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=os.environ.get("YLM_PACK", ""), required=not bool(os.environ.get("YLM_PACK")))
    ap.add_argument("--parallel", type=int, default=int(os.environ.get("YLM_PARALLEL", "2")))
    ap.add_argument("--workdir", default=os.environ.get("YLM_WORKDIR", "."))
    ap.add_argument("--extra", default=os.environ.get("YLM_EXTRA_ARGS", ""), help="extra CLI flags inserted before > redirect")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    (workdir / "runs").mkdir(parents=True, exist_ok=True)
    pack_path = Path(args.pack)
    if not pack_path.is_absolute():
        pack_path = workdir / pack_path

    cmds = [line.strip() for line in pack_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]
    if args.extra:
        cmds = [c.replace(" > ", f" {args.extra} > ", 1) if " > " in c else f"{c} {args.extra}" for c in cmds]

    if args.dry_run:
        for c in cmds:
            print(c)
        return

    pending = list(cmds)
    running: list[tuple[str, subprocess.Popen]] = []
    failed = 0
    while pending or running:
        still_running = []
        for cmd, proc in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((cmd, proc))
            elif rc != 0:
                print(f"[FAIL rc={rc}] {cmd}", flush=True)
                failed += 1
            else:
                print(f"[DONE] {cmd}", flush=True)
        running = still_running
        while pending and len(running) < max(1, args.parallel):
            cmd = pending.pop(0)
            print(f"[START] {cmd}", flush=True)
            running.append((cmd, subprocess.Popen(cmd, shell=True, cwd=str(workdir))))
            time.sleep(2.0)
        if running:
            time.sleep(5.0)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
