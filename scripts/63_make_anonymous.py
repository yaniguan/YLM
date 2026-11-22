"""Build an anonymized zip archive of the repository for double-blind review.

  python scripts/63_make_anonymous.py [--ref HEAD] [--out anonymous_YLM.zip] [--extra-terms term1,term2]

Exports tracked files from `git archive` (no `.git` history), replaces detected repository
owner/URL strings and local home paths, strips `Co-authored-by:` lines, and verifies that no
forbidden strings remain in text files. Additional sensitive strings can be supplied via
`--extra-terms` or the comma-separated `ANONYMIZE_TERMS` environment variable.
"""
import argparse
import io
import os
import re
import subprocess
import sys
import tarfile
import zipfile

DROP_SUFFIX = (".pt",)
TEXT_SUFFIX = (".py", ".md", ".txt", ".sh", ".yml", ".yaml", ".json", ".jsonl", ".tex", ".bib", ".cfg", ".toml", "LICENSE")


def detect_repo_owner() -> list[str]:
    out = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
    if out.returncode != 0:
        return []
    url = out.stdout.strip()
    m = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", url)
    return [m.group(1)] if m else []


def build_rules(extra_terms: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    owners = detect_repo_owner()
    replace = [
        (r"github\.com/[A-Za-z0-9_.-]+/YLM", "anonymous.example/YLM"),
        (r"[A-Za-z0-9_.-]+\.github\.io/YLM", "anonymous.example/YLM"),
        (r"/Users/\w+", "/home/user"),
    ]
    forbidden = [r"/Users/"]
    for owner in owners:
        esc = re.escape(owner)
        replace.append((esc, "anonymous"))
        forbidden.append(esc)
    for term in extra_terms:
        t = term.strip()
        if t:
            esc = re.escape(t)
            replace.append((esc, "anonymous"))
            forbidden.append(esc)
    return replace, forbidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--out", default="anonymous_YLM.zip")
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument(
        "--extra-terms",
        default=os.environ.get("ANONYMIZE_TERMS", ""),
        help="comma-separated additional strings to scrub and verify",
    )
    args = ap.parse_args()
    extra = [x for x in args.extra_terms.split(",") if x.strip()]
    replace, forbidden = build_rules(extra)
    raw = subprocess.run(["git", "archive", "--format=tar", args.ref], capture_output=True, check=True).stdout
    bad, n = [], 0
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar, zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            if m.name.endswith(DROP_SUFFIX) and not args.keep_checkpoints:
                continue
            data = tar.extractfile(m).read()
            name = m.name
            for pat, rep in replace:
                name = re.sub(pat, rep, name)
            if m.name.endswith(TEXT_SUFFIX) or "." not in m.name.split("/")[-1]:
                try:
                    s = data.decode("utf-8")
                except UnicodeDecodeError:
                    s = None
                if s is not None:
                    for pat, rep in replace:
                        s = re.sub(pat, rep, s)
                    s = "\n".join(line for line in s.split("\n") if not re.match(r"(?i)^co-authored-by:", line.strip()))
                    for pat in forbidden:
                        if re.search(pat, s, re.I):
                            bad.append((name, pat))
                    data = s.encode("utf-8")
            z.writestr(name, data)
            n += 1
    print(f"wrote {args.out}: {n} files")
    if bad:
        print("FORBIDDEN STRINGS STILL PRESENT (fix before releasing):")
        for name, pat in bad:
            print("  ", name, "->", pat)
        sys.exit(1)
    print("no forbidden strings found in text files")


if __name__ == "__main__":
    main()
