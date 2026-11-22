"""Build facts blocks for every condition. Layers: Q, F, S, M (+ identity), controls, placeholder."""
import itertools
import random

import features as FE
from tables import ELECTROLYTE_SOLVENTS, SOLVENTS

SYN = {
    "dmso": "dimethyl sulfoxide",
    "dmf": "dimethylformamide",
    "thf": "tetrahydrofuran",
    "dcm": "dichloromethane",
    "methylene chloride": "dichloromethane",
    "etoh": "ethanol",
    "meoh": "methanol",
    "mecn": "acetonitrile",
    "ether": "diethyl ether",
    "ipa": "isopropanol",
    "etoac": "ethyl acetate",
    "h2o": "water",
    "dioxane": "1,4-dioxane",
    "nmp": "n-methyl-2-pyrrolidone",
    "propan-2-ol": "isopropanol",
    "2-propanol": "isopropanol",
}
LAYERS = ["Q", "F", "S", "M"]

_ELEC_SMILES = {
    "EC": "O=C1OCCO1",
    "PC": "CC1COC(=O)O1",
    "DMC": "COC(=O)OC",
    "DEC": "CCOC(=O)OCC",
    "EMC": "CCOC(=O)OC",
    "DME": "COCCOC",
    "DOL": "C1COCO1",
    "THF": "C1CCOC1",
    "2-MeTHF": "CC1CCCO1",
    "2-Glyme": "COCCOCCOC",
    "3-Glyme": "COCCOCCOCCOC",
    "4-Glyme": "COCCOCCOCCOCCOC",
    "g-Butyrolactone": "O=C1CCCO1",
    "AN": "CC#N",
    "EA": "CCOC(C)=O",
    "DMSO": "CS(C)=O",
}
_ELEC_BY_SMI = {FE.canon(smi): key for key, smi in _ELEC_SMILES.items() if FE.canon(smi)}


def solvent_key(name):
    if not name:
        return None
    n = name.lower().strip()
    n = SYN.get(n, n)
    for k in SOLVENTS:
        if n == k or n.startswith(k):
            return k
    return None


def ctx_of(item):
    c = dict(item.get("context") or {})
    c["solvent_key"] = solvent_key(c.get("solvent"))
    for x in c.get("concentrations") or []:
        if "smiles" not in x:
            for m in item.get("molecules", []):
                if m["name"].lower() == str(x.get("species", "")).lower():
                    x["smiles"] = m["smiles"]
    return c


def electrolyte_lines(item, layer, shuffled=False):
    """Xu (2004) battery-solvent values; when shuffled=True, look up by substituted SMILES only."""
    out = []
    for m in item.get("molecules", []):
        c_smi = FE.canon(m.get("smiles", "")) if m.get("smiles") else None
        key = _ELEC_BY_SMI.get(c_smi) if c_smi else None
        if key is None and not shuffled:
            key = m["name"]
        v = ELECTROLYTE_SOLVENTS.get(key) if key else None
        if not v:
            continue
        e, eta, mp, bp = v
        known = FE.SOLV_BY_SMI.get(c_smi) is not None if c_smi else False
        if layer == "S":
            out.append(f"- {m['name']} (as electrolyte solvent): viscosity {eta:g} cP" + ("" if known else f", dielectric {e:g}"))
        if layer == "M" and not known:
            out.append(f"- {m['name']}: mp {mp:g} C, bp {bp:g} C")
    return out


def layer_block(layer, item, fm_pred, smap=None):
    mols = item.get("molecules", [])
    if smap:
        mols = [{"name": m["name"], "smiles": smap.get(m["smiles"], m["smiles"])} for m in mols]
    ctx = ctx_of(item)
    if layer == "Q":
        s = FE.block_Q(mols, ctx)
    elif layer == "F":
        s = FE.block_F(mols, ctx, fm_pred)
    elif layer == "S":
        s = FE.block_S(mols, ctx)
    elif layer == "M":
        s = FE.block_M(mols, ctx)
    ex = electrolyte_lines({"molecules": mols}, layer, shuffled=bool(smap))
    if ex:
        s += "\n" + "\n".join(ex)
    return s


def identity_block(item):
    mols = item.get("molecules", [])
    return (
        "Molecules:\n" + "\n".join(f"- {m['name']}: SMILES {m['smiles']}" for m in mols)
        if mols
        else "Molecules: none listed"
    )


def make_smap(items, seed=0):
    """Derangement of SMILES within a benchmark: each molecule's facts come from a different molecule."""
    smis = sorted({m["smiles"] for it in items for m in it.get("molecules", [])})
    rng = random.Random(seed)
    perm = smis[:]
    for _ in range(100):
        rng.shuffle(perm)
        if all(a != b for a, b in zip(smis, perm)):
            break
    return dict(zip(smis, perm))


def facts(item, cond, fm_pred, smap):
    """cond: dict(layers=set, identity=bool, shuffled=bool, placeholder=bool)"""
    parts = []
    if cond.get("identity", True):
        parts.append(identity_block(item))
    if cond.get("placeholder"):
        target = len("\n\n".join(layer_block(layer, item, fm_pred) for layer in LAYERS))
        parts.append(FE.block_placeholder(item.get("molecules", []), target))
    for layer in LAYERS:
        if layer in cond.get("layers", set()):
            parts.append(layer_block(layer, item, fm_pred, smap if cond.get("shuffled") else None))
    return "\n\n".join(parts)


def factorial_conditions():
    conds = {}
    for r in range(5):
        for sub in itertools.combinations(LAYERS, r):
            name = "B" if not sub else "+" + "".join(sub)
            conds[name] = {"layers": set(sub), "identity": True}
    conds["A"] = {"layers": set(), "identity": False}
    conds["ALL-shuf"] = {"layers": set(LAYERS), "identity": True, "shuffled": True}
    conds["ALL-plac"] = {"layers": set(LAYERS), "identity": True, "placeholder": True}
    return conds
