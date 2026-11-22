"""Token factory v1 (issue #80): make new, checked force questions on train molecules.

Design: round2 "Revised design", sections 1-3, 7 and 9.

An item is dict(mol=label index, atom=atom index, question=str, arm=str).

Three proposer arms:
  A  every atom that 02's unique_atom_descriptions can describe x the 4 train force frames (no LLM)
  B  a teacher LM writes items from an atom table (prompt + parser are here, the LM call is in the script)
  C  mutated items that a correct verifier must reject (verifier test only, never kept)

Verifiers run in order and stop at the first failure. Each returns (ok, reason):
  V0  format: keys, types, length
  V1  split: mol is a loaded train molecule
  V2a vocab: the question is a train frame around a description in 02's closed grammar
  V2b exactly one "the <element> atom" phrase and at most one bonded / no-bonds clause
      (redundant backstop: V2a's grammar already allows only one of each, so V2b can not fire
      after V2a passes; it is kept in case V2a's grammar is loosened later)
  V2c resolve(question, mol) == atom (exact reading)
  V2d the "at least" reading picks the same single atom
  V3  label: the xTB force on the atom exists and is finite

After the verifiers, an item is dropped if it repeats an earlier item of the run (stage "dedup")
or a TensorQA-v0 question on the same (mol, atom) (stage "dedup_v0").

The answer of a kept item is the xTB force on the atom (hartree/bohr), looked up from the labels.
V2a accepts only 02's grammar, letter case included, so the factory can not add new ways to describe atoms.
"""
import hashlib
import importlib.util
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.parse_question import COUNT, ELEMENT, parse, resolve

VERIFIER_VERSION = "tf-v1.1"

# same value as src.provenance.OFFICIAL_V0_UNPACKED_JSON_SHA256 (checked in tests); copied here so
# arms A and C do not import torch / e3nn through src.provenance
OFFICIAL_V0_UNPACKED_JSON_SHA256 = "76cf64ea621b5c5e521caeb1bbc6511bfb4b0fac7a521e79d67f5249a73630d3"

NAME = {1: "hydrogen", 6: "carbon", 7: "nitrogen", 8: "oxygen", 9: "fluorine"}
NAME_PLURAL = {1: "hydrogens", 6: "carbons", 7: "nitrogens", 8: "oxygens", 9: "fluorines"}
COUNT_WORD = {1: "one", 2: "two", 3: "three", 4: "four"}
FORCE_TEMPLATES = [
    "What is the force on {d}?",
    "Give the force vector acting on {d}.",
    "Predict the force felt by {d}.",
    "How is {d} being pushed?",
]
FORCE_TEMPLATES_HELDOUT = [
    "Report the net force on {d}.",
    "Which direction is {d} pulled, and how hard?",
]


def describe_neighbors(atom):
    counts = Counter(n.GetAtomicNum() for n in atom.GetNeighbors())
    parts = []
    for z in sorted(counts):
        k = counts[z]
        word = NAME[z] if k == 1 else NAME_PLURAL[z]
        parts.append(f"{COUNT_WORD.get(k, str(k))} {word}")
    if not parts:
        return "with no bonds"
    if len(parts) == 1:
        return "bonded to " + parts[0]
    return "bonded to " + ", ".join(parts[:-1]) + " and " + parts[-1]


def unique_atom_descriptions(mol):
    atoms = list(mol.GetAtoms())
    short = {a.GetIdx(): f"the {NAME[a.GetAtomicNum()]} atom {describe_neighbors(a)}" for a in atoms}
    counts = Counter(short.values())
    result = {a.GetIdx(): short[a.GetIdx()] for a in atoms if counts[short[a.GetIdx()]] == 1}
    ring = {}
    for a in atoms:
        i = a.GetIdx()
        if i in result:
            continue
        where = "in a ring" if a.IsInRing() else "not in a ring"
        ring[i] = f"the {NAME[a.GetAtomicNum()]} atom {where} {describe_neighbors(a)}"
    ring_counts = Counter(ring.values())
    all_long = Counter(list(ring.values()) + list(result.values()))
    for i, d in ring.items():
        if ring_counts[d] == 1 and all_long[d] == 1:
            result[i] = d
    return result

SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

# closed grammar of 02's descriptions
_EL = "(?:" + "|".join(NAME.values()) + ")"
_ELS = "(?:" + "|".join(NAME_PLURAL.values()) + ")"
_GROUP = rf"(?:one {_EL}|(?:two|three|four) {_ELS})"
_LIST = rf"{_GROUP}(?:(?:, {_GROUP})* and {_GROUP})?"
DESCRIPTION = re.compile(rf"the {_EL} atom(?: in a ring| not in a ring)? (?:bonded to ({_LIST})|with no bonds)")
GROUP_ITEM = re.compile("(" + "|".join(COUNT) + rf") ({_EL})")
HEADS = re.compile(rf"\bthe {_EL} atoms?\b")
CLAUSES = re.compile(r"bonded to|with no bonds")

# which verifier is the first to reject each mutation kind (arm C)
EXPECTED_REJECT = {
    "negation": "V2a",     # "not bonded to": "not" only allowed in "not in a ring"
    "article": "V2a",      # "bonded to a nitrogen": no article in the grammar (resolve alone would accept)
    "or": "V2a",           # "or" lists are not in the grammar
    "two_atoms": "V2a",    # two atom phrases (V2b would also catch it)
    "plural": "V2a",       # "the oxygen atoms" (V2b would also catch it)
    "partial": "V2c",      # grammatical, but the exact reading never picks the true atom
    "wrong_atom": "V2c",   # the description is unique, so it resolves to the original atom
    "heldout": "V2a",      # held-out frames must never be produced
    "extra_words": "V2a",  # "next to the carbonyl" is not in the grammar
    "uppercase": "V2a",    # "THE CARBON ATOM ...": the grammar is lowercase only
    "reorder": "V2a",      # "one carbon and three hydrogens": resolve alone would accept
}


def file_sha256(path):
    """Hex SHA-256 of the file at path, or None if path is unset or missing."""
    if not path or not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------- small text helpers ----------

def neighbor_parts(counts):
    """Counter {Z: k} -> ['three hydrogens', 'one carbon'] in 02's order (H, C, N, O, F)."""
    return [f"{COUNT_WORD.get(counts[z], str(counts[z]))} {NAME[z] if counts[z] == 1 else NAME_PLURAL[z]}"
            for z in sorted(counts) if counts[z] > 0]


def join_parts(parts, last=" and "):
    """Same joining as 02's describe_neighbors."""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + last + parts[-1]


def split_frame(question, templates=FORCE_TEMPLATES):
    """-> (template, description), or (None, None) if no frame fits exactly. Case-sensitive."""
    for t in templates:
        pre, post = t.split("{d}")
        if len(question) > len(pre) + len(post) and question.startswith(pre) and question.endswith(post):
            return t, question[len(pre):len(question) - len(post)]
    return None, None


def normalise(question):
    return " ".join(question.lower().split())


def atleast_atoms(spec, rdkit_mol):
    """Atoms of the same element (and ring flag if given) with at least the listed neighbours.
    'with no bonds' (empty Counter) is read exactly."""
    hits = []
    for a in rdkit_mol.GetAtoms():
        if a.GetAtomicNum() != spec["element"]:
            continue
        if spec["ring"] is not None and a.IsInRing() != spec["ring"]:
            continue
        nb = Counter(n.GetAtomicNum() for n in a.GetNeighbors())
        want = spec["neighbors"]
        if want is not None:
            if not want and nb:
                continue
            if any(nb[z] < k for z, k in want.items()):
                continue
        hits.append(a.GetIdx())
    return hits


# ---------- proposers ----------

def propose_templates(mol_index, rdkit_mol):
    """Arm A: every uniquely describable atom x the 4 train force frames."""
    items = []
    for atom, d in sorted(unique_atom_descriptions(rdkit_mol).items()):
        for t in FORCE_TEMPLATES:
            items.append(dict(mol=mol_index, atom=atom, question=t.format(d=d), arm="A"))
    return items


def mutate(item, rdkit_mol, rng):
    """Arm C: wrong or forbidden variants of a valid arm-A item, one per kind where it applies.
    See EXPECTED_REJECT for the verifier that should catch each kind."""
    frame, desc = split_frame(item["question"])
    m = re.fullmatch(r"(the \w+ atom(?: in a ring| not in a ring)?) (.*)", desc or "")
    if m is None:
        return []
    head, clause = m.groups()
    atom = item["atom"]
    counts = Counter(n.GetAtomicNum() for n in rdkit_mol.GetAtomWithIdx(atom).GetNeighbors())
    parts = neighbor_parts(counts)
    out = {}

    def ask(d, f=frame):
        return f.format(d=d)

    out["heldout"] = ask(desc, rng.choice(FORCE_TEMPLATES_HELDOUT))
    out["extra_words"] = ask(desc + " next to the carbonyl")
    out["plural"] = ask(head.replace(" atom", " atoms", 1) + " " + clause)
    out["uppercase"] = ask(desc.upper())
    if parts:
        out["negation"] = ask(f"{head} not {clause}")
        z0 = min(counts)
        out["article"] = ask(f"{head} bonded to " + join_parts([f"a {NAME[z0]}"] + parts[1:]))
        if len(parts) >= 2:
            out["or"] = ask(f"{head} bonded to " + join_parts(parts, last=" or "))
            out["reorder"] = ask(f"{head} bonded to " + join_parts(parts[::-1]))
        else:
            other = rng.choice([z for z in NAME if z not in counts])
            out["or"] = ask(f"{head} bonded to {parts[0]} or one {NAME[other]}")
        fewer = counts.copy()
        fewer[rng.choice(sorted(counts))] -= 1
        if sum(fewer.values()) > 0:
            out["partial"] = ask(f"{head} bonded to " + join_parts(neighbor_parts(fewer)))
    descriptions = unique_atom_descriptions(rdkit_mol)
    others = [i for i in sorted(descriptions) if i != atom]
    if others:
        out["two_atoms"] = ask(desc + " and " + descriptions[rng.choice(others)])
    variants = [dict(item, question=q, arm="C", mutation=kind) for kind, q in out.items()]
    wrong = [a.GetIdx() for a in rdkit_mol.GetAtoms() if a.GetIdx() != atom]
    if wrong:
        variants.append(dict(item, atom=rng.choice(wrong), arm="C", mutation="wrong_atom"))
    return variants


def build_teacher_prompt(rdkit_mol):
    """Arm B: prompt with a numbered atom table. The teacher answers with one JSON line."""
    rows = []
    for a in rdkit_mol.GetAtoms():
        nb = Counter(n.GetAtomicNum() for n in a.GetNeighbors())
        rows.append(f"{a.GetIdx()} | {NAME[a.GetAtomicNum()]} | {', '.join(neighbor_parts(nb)) or 'none'} | "
                    f"{'yes' if a.IsInRing() else 'no'}")
    frames = "\n".join("  " + t.replace("{d}", "<description>") for t in FORCE_TEMPLATES)
    return (
        "Here is a molecule as a table of atoms.\n"
        "index | element | bonded neighbours | in a ring\n" + "\n".join(rows) + "\n\n"
        "Write ONE question that asks for the force on ONE atom of this molecule.\n"
        "Rules:\n"
        "- Use exactly one of these frames:\n" + frames + "\n"
        "- The description names the element and the FULL list of bonded neighbours, for example\n"
        "  \"the carbon atom bonded to three hydrogens and one carbon\".\n"
        "- Write counts as words: one, two, three, four. Use the plural after two, three, four.\n"
        "- List neighbours in the order hydrogen, carbon, nitrogen, oxygen, fluorine, joined with \", \" "
        "and a final \" and \".\n"
        "- If the atom has no neighbours, write \"the <element> atom with no bonds\".\n"
        "- Add \"in a ring\" or \"not in a ring\" right after \"atom\" only if it is needed so that the "
        "description fits exactly one atom.\n"
        "- One atom only. No other words.\n"
        "Answer with ONE JSON line and nothing else: {\"atom\": <index>, \"question\": \"...\"}"
    )


_JSON_OBJECT = re.compile(r"\{[^{}]*\}")


def parse_teacher_output(text, mol=None):
    """First JSON object with 'atom' and 'question' in the text -> arm-B item, else None.
    Types are not fixed here; V0 checks them."""
    for m in _JSON_OBJECT.finditer(text or ""):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(obj, dict) and "atom" in obj and "question" in obj:
            return dict(mol=mol, atom=obj["atom"], question=obj["question"], arm="B", raw=text)
    return None


# ---------- verifiers: (item, ctx) -> (ok, reason) ----------

def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def verify_format(item, ctx):
    if not isinstance(item, dict) or any(k not in item for k in ("mol", "atom", "question", "arm")):
        return False, "V0: missing keys"
    if not (_is_int(item["mol"]) and _is_int(item["atom"]) and isinstance(item["arm"], str)):
        return False, "V0: wrong types"
    if not isinstance(item["question"], str) or not 10 <= len(item["question"]) <= 200:
        return False, "V0: question not a str of 10..200 chars"
    return True, ""


def verify_split(item, ctx):
    if item["mol"] not in ctx["train_ids"]:
        return False, "V1: mol not in train split"
    if item["mol"] not in ctx["mols"]:
        return False, "V1: mol not loaded"
    return True, ""


def verify_vocab(item, ctx):
    frame, desc = split_frame(item["question"])
    if frame is None:
        return False, "V2a: frame is not a train force frame"
    m = DESCRIPTION.fullmatch(desc)
    if m is None:
        return False, "V2a: description not in grammar"
    if m.group(1):
        zs = [ELEMENT[el] for _, el in GROUP_ITEM.findall(m.group(1))]
        if zs != sorted(set(zs)):
            return False, "V2a: neighbour list not in H, C, N, O, F order"
    return True, ""


def verify_one_atom(item, ctx):
    q = item["question"].lower()
    if len(HEADS.findall(q)) != 1:
        return False, "V2b: not exactly one atom phrase"
    if len(CLAUSES.findall(q)) > 1:
        return False, "V2b: more than one bond clause"
    return True, ""


def verify_resolve(item, ctx):
    got = resolve(item["question"].lower(), ctx["mols"][item["mol"]])
    if got == -1:
        return False, "V2c: exact reading matches 0 or >1 atoms"
    if got != item["atom"]:
        return False, "V2c: exact reading picks another atom"
    return True, ""


def verify_atleast(item, ctx):
    hits = atleast_atoms(parse(item["question"].lower()), ctx["mols"][item["mol"]])
    if len(hits) != 1:
        return False, "V2d: at-least reading matches 0 or >1 atoms"
    if hits[0] != item["atom"]:
        return False, "V2d: at-least reading picks another atom"
    return True, ""


def lookup_force(labels, mol, atom):
    """V3 answer: the xTB force on the atom, hartree/bohr, as a list of 3 floats."""
    return [float(x) for x in np.asarray(labels[mol]["forces"])[atom]]


def verify_label(item, ctx):
    lab = ctx["labels"][item["mol"]]
    numbers = [a.GetAtomicNum() for a in ctx["mols"][item["mol"]].GetAtoms()]
    if list(lab["numbers"]) != numbers:
        return False, "V3: atom order differs from labels"
    forces = np.asarray(lab["forces"])
    if not 0 <= item["atom"] < len(forces):
        return False, "V3: atom index out of range"
    if not np.isfinite(forces[item["atom"]]).all():
        return False, "V3: force not finite"
    return True, ""


STAGES = [("V0", verify_format), ("V1", verify_split), ("V2a", verify_vocab), ("V2b", verify_one_atom),
          ("V2c", verify_resolve), ("V2d", verify_atleast), ("V3", verify_label)]


# ---------- the loop ----------

def describe_constraints(question):
    """-> (coverage key, constraint-count bucket) of an accepted question."""
    spec = parse(question.lower())
    ring = {None: "no ring flag", True: "in a ring", False: "not in a ring"}[spec["ring"]]
    nb = spec["neighbors"]
    pattern = "none" if nb is None else ("no bonds" if not nb else "".join(f"{SYMBOL[z]}{nb[z]}" for z in sorted(nb)))
    key = f"{NAME[spec['element']]} | {ring} | {pattern}"
    bucket = "element only" if nb is None else ("+neighbours" if spec["ring"] is None else "+neighbours +ring")
    return key, bucket


def run_factory(items, mols, labels, train_ids, existing_pairs, existing_questions=()):
    """Verify, dedup and label the items. -> (kept, rejected, stats).
    existing_questions: (mol, atom, normalise(question)) triples of TensorQA-v0; exact copies are
    rejected with stage "dedup_v0", so every kept item is a new string.
    Arm C items are never kept: a C item that passes every verifier is a verifier hole."""
    ctx = dict(mols=mols, labels=labels, train_ids=set(train_ids))
    existing_questions = set(existing_questions)
    proposals, kept_per_arm, dedup_removed, disagree = Counter(), Counter(), Counter(), Counter()
    dedup_v0_removed = Counter()
    passed = defaultdict(Counter)
    reasons = defaultdict(Counter)
    mutations = defaultdict(Counter)
    seconds = Counter()
    coverage, buckets, frames = Counter(), Counter(), Counter()
    new_pairs = defaultdict(set)
    seen = set()
    kept, rejected = [], []

    for item in items:
        arm = item.get("arm", "?") if isinstance(item, dict) else "?"
        proposals[arm] += 1
        failed = None
        for name, fn in STAGES:
            t0 = time.perf_counter()
            ok, reason = fn(item, ctx)
            seconds[name] += time.perf_counter() - t0
            if not ok:
                failed = (name, reason)
                break
            passed[arm][name] += 1
            if name == "V2b":  # both readings are well defined from here on
                q, rd = item["question"].lower(), mols[item["mol"]]
                exact_ok = resolve(q, rd) == item["atom"]
                atleast_ok = atleast_atoms(parse(q), rd) == [item["atom"]]
                disagree[arm] += exact_ok != atleast_ok
        kind = item.get("mutation") if isinstance(item, dict) else None
        if failed:
            rejected.append(dict(item, stage=failed[0], reason=failed[1]))
            reasons[arm][failed[1]] += 1
            if kind:
                mutations[kind]["caught by " + failed[0]] += 1
                mutations[kind]["as expected" if failed[0] == EXPECTED_REJECT[kind] else "other stage"] += 1
            continue
        if arm == "C":
            rejected.append(dict(item, stage="C", reason="C: mutation passed all verifiers (verifier hole)"))
            reasons[arm]["C: mutation passed all verifiers (verifier hole)"] += 1
            mutations[kind]["missed"] += 1
            continue
        t0 = time.perf_counter()
        key = (item["mol"], item["atom"], normalise(item["question"]))
        seconds["dedup"] += time.perf_counter() - t0
        if key in seen:
            dedup_removed[arm] += 1
            rejected.append(dict(item, stage="dedup", reason="dedup: exact repeat"))
            continue
        seen.add(key)
        if key in existing_questions:
            dedup_v0_removed[arm] += 1
            rejected.append(dict(item, stage="dedup_v0", reason="dedup_v0: exact copy of a TensorQA-v0 question"))
            continue
        kept.append(dict(item, force_hartree_per_bohr=lookup_force(labels, item["mol"], item["atom"])))
        kept_per_arm[arm] += 1
        if (item["mol"], item["atom"]) not in existing_pairs:
            new_pairs[arm].add((item["mol"], item["atom"]))
        cov, bucket = describe_constraints(item["question"])
        coverage[cov] += 1
        buckets[bucket] += 1
        frames[split_frame(item["question"])[0]] += 1

    stats = dict(
        verifier_version=VERIFIER_VERSION,
        proposals=dict(proposals),
        passed_after_stage={arm: {name: passed[arm][name] for name, _ in STAGES} for arm in proposals},
        rejection_reasons={arm: dict(c) for arm, c in reasons.items()},
        mutations={kind: dict(c) for kind, c in sorted(mutations.items())},
        expected_reject=EXPECTED_REJECT,
        dedup_removed=dict(dedup_removed),
        dedup_v0_removed=dict(dedup_v0_removed),
        kept=dict(kept_per_arm),
        kept_new_strings=dict(kept_per_arm),  # equal to kept: v0 copies are removed above
        yield_per_1k={arm: round(1000 * kept_per_arm[arm] / proposals[arm], 1) for arm in proposals},
        new_mol_atom_pairs={arm: len(s) for arm, s in new_pairs.items()},
        new_mol_atom_pairs_total=len(set().union(*new_pairs.values())) if new_pairs else 0,
        v2c_v2d_disagreements=dict(disagree),
        coverage=dict(sorted(coverage.items())),
        constraint_histogram=dict(buckets),
        frames=dict(frames),
        seconds_per_stage={k: round(v, 4) for k, v in seconds.items()},
    )
    return kept, rejected, stats
