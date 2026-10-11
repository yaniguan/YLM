"""Rule-based parser: force question -> atom index. Used by the pipeline baseline.

Reads element, ring flag and bonded-neighbour counts from the words (any template wording), then
finds the atoms of the molecule that match. Returns the index only if exactly one atom matches,
otherwise -1 (parse failure). It does not look at the generator in scripts/02_make_questions.py.
"""
import re
from collections import Counter

ELEMENT = {"hydrogen": 1, "carbon": 6, "nitrogen": 7, "oxygen": 8, "fluorine": 9}
COUNT = {"one": 1, "two": 2, "three": 3, "four": 4}
HEAD = re.compile(r"the (hydrogen|carbon|nitrogen|oxygen|fluorine) atom")
RING = re.compile(r"\b(not in a ring|in a ring)\b")
BOND = re.compile(r"bonded to ((?:(?:one|two|three|four|\d+) \w+(?:, | and )?)+)")
ITEM = re.compile(r"(one|two|three|four|\d+) (hydrogen|carbon|nitrogen|oxygen|fluorine)s?")


def parse(question):
    """-> dict(element=Z, ring=True/False/None, neighbors=Counter or None) or None"""
    m = HEAD.search(question)
    if not m:
        return None
    rest = question[m.end():]
    spec = {"element": ELEMENT[m.group(1)], "ring": None, "neighbors": None}
    r = RING.search(rest)
    if r:
        spec["ring"] = r.group(1) == "in a ring"
    if "with no bonds" in rest:
        spec["neighbors"] = Counter()
    else:
        b = BOND.search(rest)
        if b:
            nb = Counter()
            for n, el in ITEM.findall(b.group(1)):
                nb[ELEMENT[el]] += COUNT[n] if n in COUNT else int(n)
            spec["neighbors"] = nb
    return spec


def resolve(question, rdkit_mol):
    """-> atom index, or -1 if the parse fails or is ambiguous"""
    spec = parse(question)
    if spec is None:
        return -1
    hits = []
    for a in rdkit_mol.GetAtoms():
        if a.GetAtomicNum() != spec["element"]:
            continue
        if spec["ring"] is not None and a.IsInRing() != spec["ring"]:
            continue
        if spec["neighbors"] is not None and Counter(n.GetAtomicNum() for n in a.GetNeighbors()) != spec["neighbors"]:
            continue
        hits.append(a.GetIdx())
    return hits[0] if len(hits) == 1 else -1
