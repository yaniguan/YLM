"""Rule-based parser: force question -> atom index. Used by the pipeline baseline.

Reads element, ring flag and bonded-neighbour counts from the words (any template wording), then
finds the atoms of the molecule that match. Returns the index only if exactly one atom matches,
otherwise -1 (parse failure). It does not look at the generator in scripts/02_make_questions.py.
"""
import re
from collections import Counter

ELEMENT = {"hydrogen": 1, "carbon": 6, "nitrogen": 7, "oxygen": 8, "fluorine": 9, "sulfur": 16, "chlorine": 17}
_EL = "|".join(ELEMENT)
COUNT = {"one": 1, "two": 2, "three": 3, "four": 4}
HEAD = re.compile(rf"the ({_EL}) atom")
RING = re.compile(r"\b(not in a ring|in a ring)\b")
BOND = re.compile(r"bonded to ((?:(?:one|two|three|four|\d+) \w+(?:, | and )?)+)")
ITEM = re.compile(rf"(one|two|three|four|\d+) ({_EL})s?")


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


# ---- two-atom (compositional) questions -------------------------------------------------------------
# One extra rule per question type: this is the maintenance cost the comparison is about. Version 2 of the rules
# was written after reading failures on the TRAIN split only (v1 benchmark); held-out wording families were not used.
DIFF = re.compile(r"\b(differ\w*|subtract\w*|minus|less|remov\w*|deduct\w*|take[sn]? away|taken away|taking away|"
                  r"negative of|excess|negative term)\b|f_a\s*[\u2212-]|\(\u2212f_b\)")
PROJ = re.compile(r"\b(project\w*|axis|axial|along|resolved|parallel|portion|share|component|part)\b")
# verbs of the form "subtract X FROM Y": the atom after "from" is the minuend A
SUBTRACT_FROM = re.compile(r"\b(subtract\w*|deduct\w*|remov\w*|taken away|take[sn]? away|taking away)\b.*\bfrom\b")
KEEP_ORDER = re.compile(r"from it\b|from the first\b|subtracting from it|as the positive term")   # "subtract ... FROM IT": A is first
NEG_FIRST = re.compile(r"negative term.*positive term")
FORCE_BEFORE = re.compile(r"force[^.?!,;]{0,40}\b(on|by|of|upon|exerted on|acting on|experienced by|felt by)\s+$")


def parse_pair(question):
    """-> (task, spec_a, spec_b) with task in {"diff", "proj"}, or None. The specs are as in parse()."""
    q = question.lower()
    heads = [m.start() for m in HEAD.finditer(q)]
    if len(heads) != 2:
        return None
    task = "diff" if DIFF.search(q) else "proj" if PROJ.search(q) else None
    first, second = parse(q[heads[0]:heads[1]]), parse(q[heads[1]:])
    if task is None or first is None or second is None:
        return None
    if task == "diff":
        swap = bool(SUBTRACT_FROM.search(q)) and not KEEP_ORDER.search(q)
        swap = swap or bool(NEG_FIRST.search(q))
        if swap:
            first, second = second, first
    else:   # proj: A is the atom whose description follows "force on / acting on / ..."; otherwise mention order
        a_first = bool(FORCE_BEFORE.search(q[:heads[0]]))
        a_second = bool(FORCE_BEFORE.search(q[heads[0]:heads[1]]))
        if a_second and not a_first:
            first, second = second, first
    return task, first, second


def _match(spec, rdkit_mol):
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


def resolve_pair(question, rdkit_mol):
    """-> (task, atom_a, atom_b); task is None on a parse failure, an atom is -1 if ambiguous or missing"""
    p = parse_pair(question)
    if p is None:
        return None, -1, -1
    task, sa, sb = p
    return task, _match(sa, rdkit_mol), _match(sb, rdkit_mol)


def resolve_program(question, rdkit_mol):
    """-> (task, atom_a, atom_b) for force / diff / proj questions. task None = parse failure.
    Exactly one atom description means a force question (atom_b = -1)."""
    heads = HEAD.findall(question.lower())
    if len(heads) == 1:
        return "force", resolve(question.lower(), rdkit_mol), -1
    return resolve_pair(question, rdkit_mol)


# ---- polarizability questions (TensorQA-pol) ----------------------------------------------------------
POL_ANISO = re.compile(r"anisotrop|traceless|zero-?trace|directional dependence|deviator|non-?isotropic|off-?diagonal|\bshape\b|deviation")
POL_IND = re.compile(r"\binduc\w*|response dipole|dipole response|dipole vector|vector dipole|\bp\s*=|alpha times|\u03b1\s*[\u00b7*]|dipole (moment |vector )?(that|which|produced|generated|created|developed|arises|results|response)")
FROM_BEFORE = re.compile(r"(from|starting (at|from)|origin(ating)? (at|from)|tail at)\s+(the\s+)?$")


def resolve_pol(question, rdkit_mol):
    """-> (task, atom_a, atom_b), task in pol_iso / pol_aniso / pol_dir / pol_ind (None on failure).
    Two atom descriptions: a direction; the atom after 'from' is A, the other is B (mention order otherwise)."""
    q = question.lower()
    heads = [m.start() for m in HEAD.finditer(q)]
    if not heads:
        task = "pol_aniso" if POL_ANISO.search(q) else "pol_iso" if re.search(r"isotrop|average|mean|orientation|rotation|trace|scalar|one-third|third", q) else None
        return task, -1, -1
    if len(heads) != 2:
        return None, -1, -1
    first, second = parse(q[heads[0]:heads[1]]), parse(q[heads[1]:])
    if first is None or second is None:
        return None, -1, -1
    a_second = bool(FROM_BEFORE.search(q[:heads[1]].rstrip() + " ")) and not bool(FROM_BEFORE.search(q[:heads[0]].rstrip() + " "))
    if a_second:
        first, second = second, first
    return ("pol_ind" if POL_IND.search(q) else "pol_dir"), _match(first, rdkit_mol), _match(second, rdkit_mol)
