"""Render per-item information layers (Q, F, S, M) and controls as text facts blocks.
Facts are computed from SMILES and stated conditions only; they never see the question."""
import json, math, hashlib, random, numpy as np
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from tables import SOLVENTS, SOLVENT_SMILES, PKA
RDLogger.DisableLog("rdApp.*")
from paths import ROOT, CACHE
HERE = ROOT
HARTREE_EV, HARTREE_KCAL = 27.211386, 627.5095
ALPB = {"water": "water", "methanol": "methanol", "ethanol": "ethanol", "acetone": "acetone",
        "acetonitrile": "acetonitrile", "dimethyl sulfoxide": "dmso", "dimethylformamide": "dmf",
        "tetrahydrofuran": "thf", "diethyl ether": "ether", "dichloromethane": "ch2cl2",
        "chloroform": "chcl3", "toluene": "toluene", "hexane": "hexane", "ethyl acetate": "ethylacetate",
        "benzene": "benzene", "1,4-dioxane": "dioxane"}
HAZ = {"acyl halide": "C(=O)[Cl,Br,I,F]", "acid anhydride": "C(=O)OC(=O)", "isocyanate": "N=C=O",
       "azide": "[N-]=[N+]=[N-]", "organic azide": "N=[N+]=[N-]", "peroxide O-O": "OO",
       "ether with alpha C-H": "[CX4;!H0]O[CX4]", "aldehyde": "[CX3H1](=O)[#6]", "epoxide": "C1OC1",
       "michael acceptor C=C-C=O": "C=CC=O", "thiol": "[SX2H]", "nitro": "[N+](=O)[O-]",
       "hydride anion": "[H-]", "complex hydride": "[AlH4-,BH4-,BH3-]", "carbon-metal bond": "[#6][Li,Mg,Zn]",
       "hypochlorite": "[O-]Cl", "perchlorate": "Cl(=O)(=O)(=O)[O-]", "nitrate": "[N+](=O)([O-])[O-]",
       "permanganate": "[Mn](=O)(=O)(=O)[O-]", "maleimide": "O=C1C=CC(=O)N1", "nhs ester": "C(=O)ON1C(=O)CCC1=O",
       "alkali metal (elemental)": "[Li,Na,K;X0;+0]", "primary amine": "[NX3;H2][CX4]",
       "carboxylic acid": "C(=O)[OH]", "sulfonyl halide": "S(=O)(=O)[Cl,F]", "cyanide": "[C-]#N",
       "isothiocyanate": "N=C=S", "chlorinated alkane": "[CX4]Cl"}
HAZ_P = {k: Chem.MolFromSmarts(v) for k, v in HAZ.items()}

def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

def parent(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    m = rdMolStandardize.LargestFragmentChooser().choose(m)
    m = rdMolStandardize.Uncharger().uncharge(m)
    return Chem.MolToSmiles(m)

PKA_BY_SMI = {}
for n, (s, p, d) in PKA.items():
    c = parent(s)
    if c: PKA_BY_SMI.setdefault(c, (n, p, d))
SOLV_BY_SMI = {canon(s): n for n, s in SOLVENT_SMILES.items()}

def _cache(kind, smi):
    return CACHE / f"{kind}_{hashlib.md5(smi.encode()).hexdigest()}.json"

def xtb_frag(smi, solvent=None):
    """GFN2-xTB single points on an MMFF/UFF conformer of one fragment."""
    from tblite.interface import Calculator
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    if m.GetNumAtoms() > 90: return None
    q = Chem.GetFormalCharge(m)
    if m.GetNumAtoms() == 1:
        pos = np.zeros((1, 3))
    else:
        if AllChem.EmbedMolecule(m, randomSeed=7) != 0:
            if AllChem.EmbedMolecule(m, randomSeed=7, useRandomCoords=True) != 0: return None
        try:
            if AllChem.MMFFHasAllMoleculeParams(m): AllChem.MMFFOptimizeMolecule(m, maxIters=500)
            else: AllChem.UFFOptimizeMolecule(m, maxIters=500)
        except Exception: pass
        pos = m.GetConformer().GetPositions()
    z = np.array([a.GetAtomicNum() for a in m.GetAtoms()]); x = pos / 0.52917721
    nel = int(z.sum() - q)
    def sp(charge, uhf, solv=None):
        c = Calculator("GFN2-xTB", z, x, charge=charge, uhf=uhf); c.set("verbosity", 0)
        if solv: c.add("alpb-solvation", solv)
        return c.singlepoint()
    out = {}
    r = sp(q, nel % 2); e0 = r.get("energy")
    occ = r.get("orbital-occupations"); eps = r.get("orbital-energies") * HARTREE_EV
    homo = eps[occ > 0.5].max() if (occ > 0.5).any() else None
    lumo = eps[occ < 0.5].min() if (occ < 0.5).any() else None
    ch = r.get("charges"); dip = float(np.linalg.norm(r.get("dipole"))) * 2.541746
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    out.update(homo_eV=homo, lumo_eV=lumo, gap_eV=(lumo - homo) if homo is not None and lumo is not None else None,
               dipole_D=dip, qmin=(float(ch.min()), sym[int(ch.argmin())]), qmax=(float(ch.max()), sym[int(ch.argmax())]))
    try: out["vIP_eV"] = (sp(q + 1, (nel - 1) % 2).get("energy") - e0) * HARTREE_EV
    except Exception: out["vIP_eV"] = None
    try: out["vEA_eV"] = (e0 - sp(q - 1, (nel + 1) % 2).get("energy")) * HARTREE_EV
    except Exception: out["vEA_eV"] = None
    for s in {"water", solvent} - {None}:
        if s in ALPB:
            try: out[f"dGsolv_{s}_kcal"] = (sp(q, nel % 2, ALPB[s]).get("energy") - e0) * HARTREE_KCAL
            except Exception: pass
    return out

def qm_and_solv(smi, solvent=None):
    key = f"{smi}|{solvent}"; f = _cache("xtb", key)
    if f.exists(): return json.loads(f.read_text())
    res = {}
    for frag in sorted(set(smi.split("."))):
        try: res[frag] = xtb_frag(frag, solvent)
        except Exception as e: res[frag] = None
    f.write_text(json.dumps(res)); return res

def rdkit_desc(smi):
    m = Chem.MolFromSmiles(smi)
    return dict(logP=Crippen.MolLogP(m), TPSA=rdMolDescriptors.CalcTPSA(m), HBD=rdMolDescriptors.CalcNumHBD(m),
                HBA=rdMolDescriptors.CalcNumHBA(m), MW=Descriptors.MolWt(m),
                hazards=[k for k, p in HAZ_P.items() if p is not None and m.HasSubstructMatch(p)])

def fmt(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"

def block_Q(mols, ctx):
    lines = []
    for mo in mols:
        r = qm_and_solv(mo["smiles"], (ctx or {}).get("solvent_key"))
        for frag, d in r.items():
            if not d: lines.append(f"- {mo['name']} [{frag}]: xTB not computed"); continue
            lines.append(f"- {mo['name']} [{frag}]: HOMO {fmt(d['homo_eV'])} eV, LUMO {fmt(d['lumo_eV'])} eV, "
                         f"gap {fmt(d['gap_eV'])} eV, dipole {fmt(d['dipole_D'],1)} D, vertical IP {fmt(d['vIP_eV'])} eV, "
                         f"vertical EA {fmt(d['vEA_eV'])} eV, most negative atom {d['qmin'][1]} ({d['qmin'][0]:+.2f} e), "
                         f"most positive atom {d['qmax'][1]} ({d['qmax'][0]:+.2f} e)")
    return "First-principles (GFN2-xTB, gas phase, single conformer):\n" + "\n".join(lines)

def block_F(mols, ctx, fm_pred):
    lines = []
    for mo in mols:
        d = rdkit_desc(mo["smiles"]); p = fm_pred.get(mo["smiles"], {})
        lines.append(f"- {mo['name']}: predicted logS {fmt(p.get('logS'),1)} (log mol/L), predicted logD7.4 "
                     f"{fmt(p.get('logD'),1)}, Crippen logP {d['logP']:.1f}, TPSA {d['TPSA']:.0f} A^2, HBD {d['HBD']}, "
                     f"HBA {d['HBA']}, substructures: {', '.join(d['hazards']) or 'none flagged'}")
    return "Molecular model features (MoLFormer-XL heads + RDKit):\n" + "\n".join(lines)

def block_S(mols, ctx):
    lines = []; solv = (ctx or {}).get("solvent_key")
    for mo in mols:
        r = qm_and_solv(mo["smiles"], solv); parts = []
        for frag, d in r.items():
            if not d: continue
            gs = [f"{k.split('_')[1]} {v:.1f}" for k, v in d.items() if k.startswith("dGsolv_")]
            if gs: parts.append(f"[{frag}] ALPB solvation free energy (kcal/mol): " + ", ".join(gs))
        nm = SOLV_BY_SMI.get(canon(mo["smiles"]))
        if nm:
            e, bp, mp, prot = SOLVENTS[nm]
            parts.append(f"as solvent: dielectric {e}, {'protic' if prot else 'aprotic'}")
        lines.append(f"- {mo['name']}: " + ("; ".join(parts) if parts else "no solvation data"))
    if solv and solv in SOLVENTS:
        e, bp, mp, prot = SOLVENTS[solv]
        lines.append(f"- stated solvent {solv}: dielectric {e}, {'protic' if prot else 'aprotic'}")
    return "Solvation / electrolyte:\n" + "\n".join(lines)

def ionic_strength(concs):
    I = 0.0; got = False
    for c in concs or []:
        m = Chem.MolFromSmiles(c.get("smiles") or "")
        if m is None or c.get("M") is None: continue
        frags = Chem.GetMolFrags(m, asMols=True)
        if len(frags) < 2: continue
        got = True
        for fr in frags: I += 0.5 * c["M"] * Chem.GetFormalCharge(fr) ** 2
    return I if got else None

def block_M(mols, ctx):
    ctx = ctx or {}; lines = []
    st = []
    if ctx.get("pH") is not None: st.append(f"pH {ctx['pH']}")
    if ctx.get("T_C") is not None: st.append(f"T {ctx['T_C']} C")
    if st: lines.append("- stated conditions: " + ", ".join(st))
    I = ionic_strength(ctx.get("concentrations"))
    if I is not None: lines.append(f"- ionic strength from listed salts: {I:.3g} M")
    for mo in mols:
        p = PKA_BY_SMI.get(parent(mo["smiles"]) or "")
        parts = []
        if p:
            parts.append("aqueous pKa " + ", ".join(f"{v:g}" for v in p[1]))
            if p[2] is not None: parts.append(f"dpKa/dT {p[2]:+g} per K")
        nm = SOLV_BY_SMI.get(canon(mo["smiles"]))
        if nm:
            e, bp, mp, prot = SOLVENTS[nm]; parts.append(f"bp {bp:g} C, mp {mp:g} C")
        if parts: lines.append(f"- {mo['name']}: " + "; ".join(parts))
    return "Microenvironment:\n" + ("\n".join(lines) if lines else "- no data")

def block_placeholder(mols, target_len):
    lines = []
    for mo in mols:
        m = Chem.MolFromSmiles(mo["smiles"])
        lines.append(f"- {mo['name']}: heavy atoms {m.GetNumHeavyAtoms()}, rings {rdMolDescriptors.CalcNumRings(m)}, "
                     f"rotatable bonds {rdMolDescriptors.CalcNumRotatableBonds(m)}, stereocenters "
                     f"{len(Chem.FindMolChiralCenters(m, includeUnassigned=True))}, InChIKey prefix "
                     f"{Chem.MolToInchiKey(m)[:6]}")
    s = "Molecule record metadata:\n" + "\n".join(lines)
    k = 0
    while len(s) < target_len and mols:
        mo = mols[k % len(mols)]; m = Chem.MolFromSmiles(mo["smiles"])
        s += f"\n- {mo['name']}: record field {k+1}, atoms incl. H {Chem.AddHs(m).GetNumAtoms()}, bonds {m.GetNumBonds()}"
        k += 1
    return s
