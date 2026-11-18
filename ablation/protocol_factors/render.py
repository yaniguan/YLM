"""Build facts blocks for every condition. Layers: Q, F, S, M (+ identity), controls, placeholder."""
import json, random, itertools
from pathlib import Path
import features as FE
from tables import ELECTROLYTE_SOLVENTS, SOLVENTS
SYN = {"dmso": "dimethyl sulfoxide", "dmf": "dimethylformamide", "thf": "tetrahydrofuran", "dcm": "dichloromethane",
       "methylene chloride": "dichloromethane", "etoh": "ethanol", "meoh": "methanol", "mecn": "acetonitrile",
       "ether": "diethyl ether", "ipa": "isopropanol", "etoac": "ethyl acetate", "h2o": "water", "dioxane": "1,4-dioxane",
       "nmp": "n-methyl-2-pyrrolidone", "propan-2-ol": "isopropanol", "2-propanol": "isopropanol"}
LAYERS = ["Q", "F", "S", "M"]

def solvent_key(name):
    if not name: return None
    n = name.lower().strip()
    n = SYN.get(n, n)
    for k in SOLVENTS:
        if n == k or n.startswith(k): return k
    return None

def ctx_of(item):
    c = dict(item.get("context") or {})
    c["solvent_key"] = solvent_key(c.get("solvent"))
    for x in c.get("concentrations") or []:
        if "smiles" not in x:
            for m in item.get("molecules", []):
                if m["name"].lower() == str(x.get("species", "")).lower(): x["smiles"] = m["smiles"]
    return c

def electrolyte_lines(item, layer):
    """Xu (2004) battery-solvent values; viscosity always, other values only if the general table lacks the solvent."""
    out = []
    for m in item.get("molecules", []):
        v = ELECTROLYTE_SOLVENTS.get(m["name"])
        if not v: continue
        e, eta, mp, bp = v
        known = FE.SOLV_BY_SMI.get(FE.canon(m["smiles"])) is not None
        if layer == "S":
            out.append(f"- {m['name']} (as electrolyte solvent): viscosity {eta:g} cP" + ("" if known else f", dielectric {e:g}"))
        if layer == "M" and not known: out.append(f"- {m['name']}: mp {mp:g} C, bp {bp:g} C")
    return out

def layer_block(layer, item, fm_pred, smap=None):
    mols = item.get("molecules", [])
    if smap: mols = [{"name": m["name"], "smiles": smap.get(m["smiles"], m["smiles"])} for m in mols]
    ctx = ctx_of(item)
    if layer == "Q": s = FE.block_Q(mols, ctx)
    elif layer == "F": s = FE.block_F(mols, ctx, fm_pred)
    elif layer == "S": s = FE.block_S(mols, ctx)
    elif layer == "M": s = FE.block_M(mols, ctx)
    ex = electrolyte_lines({"molecules": mols if not smap else _shuf_names(item, smap)}, layer)
    if ex: s += "\n" + "\n".join(ex)
    return s

def _shuf_names(item, smap):
    # electrolyte table rows follow the substituted molecule (by name of the molecule whose SMILES was swapped in)
    inv = {}
    for m in item.get("molecules", []): inv[m["smiles"]] = m["name"]
    return [{"name": m["name"], "smiles": smap.get(m["smiles"], m["smiles"])} for m in item.get("molecules", [])]

def identity_block(item):
    mols = item.get("molecules", [])
    return "Molecules:\n" + "\n".join(f"- {m['name']}: SMILES {m['smiles']}" for m in mols) if mols else "Molecules: none listed"

def make_smap(items, seed=0):
    """Derangement of SMILES within a benchmark: each molecule's facts come from a different molecule."""
    smis = sorted({m["smiles"] for it in items for m in it.get("molecules", [])})
    rng = random.Random(seed); perm = smis[:]
    for _ in range(100):
        rng.shuffle(perm)
        if all(a != b for a, b in zip(smis, perm)): break
    return dict(zip(smis, perm))

def facts(item, cond, fm_pred, smap):
    """cond: dict(layers=set, identity=bool, shuffled=bool, placeholder=bool)"""
    parts = []
    if cond.get("identity", True): parts.append(identity_block(item))
    if cond.get("placeholder"):
        target = len("\n\n".join(layer_block(l, item, fm_pred) for l in LAYERS))
        parts.append(FE.block_placeholder(item.get("molecules", []), target))
    for l in LAYERS:
        if l in cond.get("layers", set()):
            parts.append(layer_block(l, item, fm_pred, smap if cond.get("shuffled") else None))
    return "\n\n".join(parts)

def factorial_conditions():
    conds = {}
    for r in range(5):
        for sub in itertools.combinations(LAYERS, r):
            name = "B" if not sub else "+" + "".join(sub)
            conds[name] = {"layers": set(sub), "identity": True}
    conds["A"] = {"layers": set(), "identity": False}
    conds["ALL-shuf"] = {"layers": set(LAYERS), "identity": True, "shuffled": True}
    conds["ALL-plac"] = {"layers": set(), "identity": True, "placeholder": True}
    return conds
