"""Build an anonymised copy of the repository for double-blind review.

  python scripts/29_make_anonymous.py [--ref HEAD] [--out anonymous_YLM.zip]

Takes the tracked files of `--ref` (git archive: no .git history, so no author names/emails), replaces the GitHub
user/repo names, the author name and local paths, drops files that cannot be anonymised (listed below), and then
FAILS if any forbidden string is still present. Review the printed report before uploading. Do not link the public
repository from the paper; point to the anonymous archive instead.
"""
import argparse
import io
import re
import subprocess
import sys
import tarfile
import zipfile

REPLACE = [
    (r"github\.com/yaniguan/YLM", "anonymous.example/YLM"),
    (r"yaniguan2@gmail\.com", "anonymous@example.com"),
    (r"yguan1@[\w.]+", "user@cluster"),
    (r"yaniguan", "anonymous"),
    (r"YaniOS", "workspace"),
    (r"/Users/\w+", "/home/user"),
    (r"Yani Guan|Yan Guan|Yani", "Anonymous"),
    (r"hoffman2\.idre\.ucla\.edu", "cluster-a.example"),
    (r"hoffman2", "cluster_a"),
    (r"Hoffman2", "Cluster A"),
    (r"expanse\.sdsc\.edu", "cluster-b.example"),
    (r"expanse", "cluster_b"),
    (r"Expanse", "Cluster B"),
]
FORBIDDEN = [r"yaniguan", r"YaniOS", r"gmail\.com", r"yguan", r"/Users/", r"hoffman2", r"expanse"]
DROP_SUFFIX = (".pt",)         # binary checkpoints are shipped separately (they embed no names, but are large)
TEXT_SUFFIX = (".py", ".md", ".txt", ".sh", ".yml", ".yaml", ".json", ".tex", ".bib", ".cfg", ".toml", "LICENSE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--out", default="anonymous_YLM.zip")
    ap.add_argument("--keep_checkpoints", action="store_true")
    args = ap.parse_args()
    raw = subprocess.run(["git", "archive", "--format=tar", args.ref], capture_output=True, check=True).stdout
    bad, n = [], 0
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar, zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            if m.name.endswith(DROP_SUFFIX) and not args.keep_checkpoints:
                continue
            data = tar.extractfile(m).read()
            name = m.name.replace("hoffman2", "cluster_a").replace("expanse", "cluster_b")
            if m.name.endswith(TEXT_SUFFIX) or "." not in m.name.split("/")[-1]:
                try:
                    s = data.decode()
                except UnicodeDecodeError:
                    s = None
                if s is not None:
                    for pat, rep in REPLACE:
                        s = re.sub(pat, rep, s)
                    s = "\n".join(l for l in s.split("\n") if "Co-Authored-By" not in l)
                    for pat in FORBIDDEN:
                        if re.search(pat, s, re.I):
                            bad.append((name, pat))
                    data = s.encode()
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
