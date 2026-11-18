"""python -m labledger verify DIR | python -m labledger export DIR OUT.jsonl [--allow-synthetic]"""

import argparse
import sys

from .export import check_training_record, to_training, write_jsonl
from .ledger import Ledger


def main(argv=None):
    ap = argparse.ArgumentParser(prog="labledger")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="check events, hash chain, tree and raw file checksums")
    v.add_argument("dir")
    e = sub.add_parser("export", help="write training JSONL (coscientist.v1)")
    e.add_argument("dir")
    e.add_argument("out")
    e.add_argument("--allow-synthetic", action="store_true")
    e.add_argument("--sealed", default=None, help="file with one sealed record id per line")
    a = ap.parse_args(argv)
    L = Ledger(a.dir)
    if a.cmd == "verify":
        problems = L.verify()
        print("\n".join(problems) or f"ok: {len(L.events)} events, {len(L.current())} live records")
        return 1 if problems else 0
    sealed = []
    if a.sealed:
        with open(a.sealed) as f:
            sealed = [line.strip() for line in f if line.strip()]
    recs = to_training(L, allow_synthetic=a.allow_synthetic, sealed_ids=sealed)
    for r in recs:
        check_training_record(r)
    write_jsonl(recs, a.out)
    print(f"wrote {len(recs)} examples to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
