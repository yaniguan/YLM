"""Multi-seed benchmark comparing language-geometry interfaces on official QM9 GFN2-xTB forces.

Compares four interfaces mounted on the same frozen 60-epoch QM9 equivariant Encoder
(checkpoints/v0/ylm_minilm_s0.pt) and frozen MiniLM-L6-v2 text representations across 3 seeds:
  1. rule_exec       Hand-written rule parser (src/parse_question.py) + exact O(3) executor
  2. optA_cont       Single-pointer invariant cross-attention (Option A)
  3. optBC_cont      Dual-pointer + unconstrained text-regressed continuous coefficients (B/C-lite)
  4. group_prog      Neuro-symbolic discrete operator + pointer/set-mask selector + exact O(3) executor

Evaluates four non-overlapping molecule-split test regimes (all targets from official v0.1.0 GFN2-xTB):
  - rigid_single:       Standard v0 phrasings, single-atom force F_a
  - ood_synonym_single: Held-out chemical symbol/connectivity synonyms & conversational phrasings, F_a
  - ood_synonym_comp:   Held-out synonym + inverted phrasings on compositional diff (F_a - F_b) & proj
  - group_ood:          Multi-atom functional/element-class net force (sum_{i in G} F_i), held-out phrasings

Run:
  python scripts/64_interface_generalization_study.py --labels <qm9_xtb.pkl> --questions <tensorqa_v0.json>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src  # noqa: E402, F401
from e3nn import o3  # noqa: E402
from src.encoder import Encoder  # noqa: E402
from src.irreps_to_text import IrrepsToText, invariants  # noqa: E402
from src.parse_question import resolve_program  # noqa: E402
from src.prog import OPS_GROUP  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ANG_TO_BOHR = 1.8897261246
N_MAX = 29
DESC_RE = re.compile(r"(the (?:hydrogen|carbon|nitrogen|oxygen|fluorine) atom[^?.!]*)")

ELEMENTS = ("hydrogen", "carbon", "nitrogen", "oxygen", "fluorine")

TRAIN_TMPL = {
    "force": [
        "What is the force on {a}?",
        "Give the force vector acting on {a}.",
        "Predict the force felt by {a}.",
        "Compute the 3D force vector on {a}.",
        "Report the atomic force experienced by {a}.",
    ],
    "diff": [
        "What is the difference between the force on {a} and the force on {b}?",
        "Subtract the force on {b} from the force on {a}.",
        "Give the force on {a} minus the force on {b}.",
        "Compute the vector difference: force on {a} less force on {b}.",
    ],
    "proj": [
        "What is the vector projection of the force on {a} onto the direction toward {b}?",
        "Project the force on {a} onto the line pointing to {b}.",
        "Give the axial projection of the force on {a} along the line toward {b}.",
    ],
    "group": [
        "What is the net force vector on {grp}?",
        "Give the sum of forces on {grp} in this molecule.",
        "Compute the total force vector summed over {grp}.",
    ],
}

OOD_TMPL = {
    "force": [
        "Pulling force vector felt at {a}?",
        "Which 3D force vector acts at {a}?",
    ],
    "diff": [
        "Relative force vector at {a} after subtracting {b}?",
        "Force imbalance vector taking {a} minus {b}?",
    ],
    "proj": [
        "Signed axial force component at {a} pointing toward {b}?",
        "Resolve the force vector at {a} parallel to the line toward {b}.",
    ],
    "group": [
        "Vector sum of forces acting across {grp}?",
        "Combined pulling force vector over {grp}?",
    ],
}

GROUP_DEFS = [
    (
        "heteroatom",
        lambda z: z in (7, 8, 9),
        "all heteroatoms (nitrogen, oxygen, and fluorine)",
        "every heteroatom center (nitrogen, oxygen, or fluorine)",
    ),
    ("oxygen", lambda z: z == 8, "all oxygen atoms", "every oxygen center"),
    ("nitrogen", lambda z: z == 7, "all nitrogen atoms", "every nitrogen center"),
    ("carbon", lambda z: z == 6, "all carbon atoms", "every carbon center in the skeleton"),
]


class IdentityEncoder(nn.Module):
    """Pass-through wrapper when pre-extracted equivariant features x are fed directly."""

    def __init__(self, irreps_out):
        super().__init__()
        self.irreps_out = o3.Irreps(irreps_out)

    def forward(self, x_pre, pos=None, atom_mask=None):
        return x_pre


class FrozenOptA(nn.Module):
    def __init__(self, text_dim: int, irreps_out, d: int = 128, dropout: float = 0.2):
        super().__init__()
        self.irreps_out = o3.Irreps(irreps_out)
        self.dropout = dropout
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d), nn.SiLU())
        self.cross = IrrepsToText(d, self.irreps_out, "32x0e + 32x1o + 8x1e + 8x2e", option="A")
        self.token_w = nn.Linear(d, 1)
        self.readout = o3.Linear(self.cross.out_irreps, "1x1o")

    def forward(self, x, pos, atom_mask, text, text_mask):
        t_in = nn.functional.dropout(text, p=self.dropout, training=self.training)
        h = self.text_proj(t_in)
        e, _ = self.cross(h, x, atom_mask)
        w = self.token_w(h).squeeze(-1).masked_fill(~text_mask, float("-inf")).softmax(-1)
        return self.readout(torch.einsum("bt,btd->bd", w, e))


class FrozenOptBCLite(nn.Module):
    """Dual-pointer + unconstrained text-regressed continuous coefficients (c1..c4)."""

    def __init__(self, text_dim: int, irreps_out, d: int = 128, key: int = 64, dropout: float = 0.2):
        super().__init__()
        self.irreps_out = o3.Irreps(irreps_out)
        self.key = key
        self.dropout = dropout
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d), nn.SiLU())
        self.token_w = nn.ModuleList([nn.Linear(d, 1) for _ in range(2)])
        self.q = nn.ModuleList([nn.Linear(d, key) for _ in range(2)])
        self.k = nn.Sequential(nn.Linear(self.irreps_out.num_irreps, 128), nn.SiLU(), nn.Linear(128, key))
        self.value = o3.Linear(self.irreps_out, "1x1o")
        self.coef = nn.Sequential(nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, 4))

    def forward(self, x, pos, atom_mask, text, text_mask):
        t_in = nn.functional.dropout(text, p=self.dropout, training=self.training)
        h = self.text_proj(t_in)
        keys = self.k(invariants(x, self.irreps_out))
        v = self.value(x) * atom_mask[..., None]
        denom = atom_mask.sum(1).clamp(min=1)[:, None, None]
        centered = (pos - (pos * atom_mask[..., None]).sum(1, keepdim=True) / denom) * atom_mask[..., None]
        F, P, pooled = [], [], []
        for i in range(2):
            w = self.token_w[i](h).squeeze(-1).masked_fill(~text_mask, float("-inf")).softmax(-1)
            hk = torch.einsum("bt,btd->bd", w, h)
            a = (torch.einsum("bk,bnk->bn", self.q[i](hk), keys) / math.sqrt(self.key)).masked_fill(
                ~atom_mask, float("-inf")
            ).softmax(-1)
            F.append(torch.einsum("bn,bnc->bc", a, v))
            P.append(torch.einsum("bn,bnc->bc", a, centered))
            pooled.append(hk)
        cls = (h * text_mask[..., None]).sum(1) / text_mask.sum(1, keepdim=True).clamp(min=1)
        c = self.coef(torch.cat(pooled + [cls], -1))
        u = P[1] - P[0]
        u = u / (u.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        p1 = (F[0] * u).sum(-1, keepdim=True) * u
        p2 = (F[1] * u).sum(-1, keepdim=True) * u
        return c[:, 0:1] * F[0] + c[:, 1:2] * F[1] + c[:, 2:3] * p1 + c[:, 3:4] * p2


class FrozenGroupProg(nn.Module):
    """Neuro-symbolic discrete operator + pointer/set-mask selector + exact O(3) executor."""

    def __init__(self, text_dim: int, irreps_out, d: int = 128, key: int = 64, dropout: float = 0.2):
        super().__init__()
        from src.prog import GroupProgModel

        self.inner = GroupProgModel(
            text_dim=text_dim,
            d=d,
            key=key,
            encoder=IdentityEncoder(irreps_out),
            text_dropout=dropout,
        )

    def forward(self, x, pos, atom_mask, text, text_mask):
        return self.inner(x, pos, atom_mask, text, text_mask)


def synonymize_desc(desc: str) -> str:
    s = desc
    s = s.replace("not in a ring", "outside any ring").replace("in a ring", "within a ring")
    s = s.replace("bonded to", "connected to")
    for el in ELEMENTS:
        s = re.sub(rf"\bthe {el} atom\b", f"the {el} center", s)
    return re.sub(r"\s+", " ", s).strip()


def connectivity_mol(numbers, pos_bohr):
    rd = Chem.RWMol()
    for z in numbers:
        rd.AddAtom(Chem.Atom(int(z)))
    conf = Chem.Conformer(len(numbers))
    for k, p in enumerate(np.asarray(pos_bohr, dtype=float) / ANG_TO_BOHR):
        conf.SetAtomPosition(k, tuple(float(v) for v in p))
    rd.AddConformer(conf)
    mol = rd.GetMol()
    rdDetermineBonds.DetermineConnectivity(mol)
    Chem.FastFindRings(mol)
    return mol


def build_dataset(labels, v0_questions, n_train=900, n_val=180, n_test=250, seed=0):
    rng = random.Random(seed)
    atom_desc_by_mol: dict[int, dict[int, str]] = defaultdict(dict)
    split_by_mol: dict[int, str] = {}
    for q in v0_questions:
        if q["split"] == "test_heldout":
            continue
        m = int(q["mol"])
        split_by_mol[m] = q["split"]
        if q["task"] == "force":
            match = DESC_RE.search(q["question"])
            if match:
                atom_desc_by_mol[m][int(q["atom"])] = match.group(1).strip()

    pools = {"train": [], "val": [], "test": []}
    for m, sp in sorted(split_by_mol.items()):
        if sp in pools and len(atom_desc_by_mol[m]) >= 2:
            pools[sp].append(m)
    for sp in pools:
        rng.shuffle(pools[sp])
    def has_complete_group(old_m: int) -> bool:
        lab = labels[old_m]
        Z_m = [int(z) for z in lab["numbers"]]
        f_m = np.asarray(lab["forces"], dtype=np.float64)
        for _, pred_fn, _, _ in GROUP_DEFS:
            cls_idx = [i for i, z in enumerate(Z_m) if pred_fn(z)]
            if len(cls_idx) >= 2 and all(np.isfinite(f_m[i]).all() for i in cls_idx):
                return True
        return False

    def select_split_mols(pool: list[int], k: int, min_group: int) -> list[int]:
        grp = [m for m in pool if has_complete_group(m)]
        other = [m for m in pool if not has_complete_group(m)]
        take_g = grp[:min(len(grp), min_group)]
        rem = (grp[len(take_g):] + other)[:max(0, k - len(take_g))]
        out = take_g + rem
        rng.shuffle(out)
        return out

    chosen_mols = (
        select_split_mols(pools["train"], n_train, min_group=350)
        + select_split_mols(pools["val"], n_val, min_group=70)
        + select_split_mols(pools["test"], n_test, min_group=120)
    )
    mol_remap = {old_m: new_m for new_m, old_m in enumerate(chosen_mols)}

    records = {"train": [], "val": [], "rigid_single": [], "ood_synonym_single": [], "ood_synonym_comp": [], "group_ood": []}

    def make_item(m_new, task_name, q_str, target, a=-1, b=-1, gmask=None):
        gm = np.zeros(N_MAX, dtype=np.float32)
        if gmask is not None:
            for idx in gmask:
                gm[idx] = 1.0
        return {
            "mol": m_new,
            "task": OPS_GROUP.index(task_name),
            "task_name": task_name,
            "question": q_str,
            "target": np.asarray(target, dtype=np.float32),
            "atoms": [int(a), int(b)],
            "group_mask": gm,
        }

    for old_m in chosen_mols:
        m_new = mol_remap[old_m]
        sp = split_by_mol[old_m]
        lab = labels[old_m]
        Z = [int(z) for z in lab["numbers"]]
        pos = np.asarray(lab["pos"], dtype=np.float64)
        forces = np.asarray(lab["forces"], dtype=np.float64)
        descs = atom_desc_by_mol[old_m]
        valid_atoms = [a for a in sorted(descs) if np.isfinite(forces[a]).all()]
        if len(valid_atoms) < 2:
            continue

        # Single-atom force
        a0 = rng.choice(valid_atoms)
        if np.linalg.norm(forces[a0]) > 1e-3:
            if sp in ("train", "val"):
                q_tr = rng.choice(TRAIN_TMPL["force"]).format(a=descs[a0])
                records[sp].append(make_item(m_new, "force", q_tr, forces[a0], a=a0))
            else:
                q_id = rng.choice(TRAIN_TMPL["force"]).format(a=descs[a0])
                q_ood = rng.choice(OOD_TMPL["force"]).format(a=synonymize_desc(descs[a0]))
                records["rigid_single"].append(make_item(m_new, "force", q_id, forces[a0], a=a0))
                records["ood_synonym_single"].append(make_item(m_new, "force", q_ood, forces[a0], a=a0))

        # Two-atom diff & proj
        a1, b1 = rng.sample(valid_atoms, 2)
        diff_vec = forces[a1] - forces[b1]
        u = pos[b1] - pos[a1]
        u = u / max(np.linalg.norm(u), 1e-8)
        proj_vec = (forces[a1] @ u) * u
        for tname, tvec in (("diff", diff_vec), ("proj", proj_vec)):
            if np.linalg.norm(tvec) <= 1e-3:
                continue
            if sp in ("train", "val"):
                q_tr = rng.choice(TRAIN_TMPL[tname]).format(a=descs[a1], b=descs[b1])
                records[sp].append(make_item(m_new, tname, q_tr, tvec, a=a1, b=b1))
            else:
                q_ood = rng.choice(OOD_TMPL[tname]).format(
                    a=synonymize_desc(descs[a1]),
                    b=synonymize_desc(descs[b1]),
                )
                records["ood_synonym_comp"].append(make_item(m_new, tname, q_ood, tvec, a=a1, b=b1))

        # Exact multi-atom group net force (when all atoms of the class in the molecule have finite xTB forces)
        n_grp_added = 0
        for _, pred_fn, grp_train, grp_ood in GROUP_DEFS:
            all_cls = [idx for idx, z in enumerate(Z) if pred_fn(z)]
            if len(all_cls) >= 2 and all(np.isfinite(forces[idx]).all() for idx in all_cls):
                g_vec = forces[all_cls].sum(axis=0)
                if np.linalg.norm(g_vec) > 2e-3:
                    if sp in ("train", "val"):
                        q_tr = rng.choice(TRAIN_TMPL["group"]).format(grp=grp_train)
                        records[sp].append(make_item(m_new, "group", q_tr, g_vec, gmask=all_cls))
                    else:
                        q_ood = rng.choice(OOD_TMPL["group"]).format(grp=grp_ood)
                        records["group_ood"].append(make_item(m_new, "group", q_ood, g_vec, gmask=all_cls))
                    n_grp_added += 1
                    if sp not in ("train", "val") or n_grp_added >= 2:
                        break

    return chosen_mols, records


@torch.no_grad()
def encode_texts(unique_strings: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 128):
    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name).eval()
    states = {}
    for i in range(0, len(unique_strings), batch_size):
        chunk = unique_strings[i:i + batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=64, return_tensors="pt")
        out = lm(**enc).last_hidden_state
        for j, s in enumerate(chunk):
            length = int(enc["attention_mask"][j].sum().item())
            states[s] = out[j, :length].cpu()
    max_t = max(v.shape[0] for v in states.values())
    dim = int(lm.config.hidden_size)
    t_pad = torch.zeros(len(unique_strings), max_t, dim)
    t_mask = torch.zeros(len(unique_strings), max_t, dtype=torch.bool)
    sid = {}
    for k, s in enumerate(unique_strings):
        st = states[s]
        t_pad[k, :st.shape[0]] = st
        t_mask[k, :st.shape[0]] = True
        sid[s] = k
    return t_pad, t_mask, sid, dim


def pack_split(items: list[dict], sid: dict[str, int]):
    return {
        "mol": torch.tensor([it["mol"] for it in items], dtype=torch.long),
        "uid": torch.tensor([sid[it["question"]] for it in items], dtype=torch.long),
        "task": torch.tensor([it["task"] for it in items], dtype=torch.long),
        "atoms": torch.tensor([it["atoms"] for it in items], dtype=torch.long),
        "group_mask": torch.tensor(np.stack([it["group_mask"] for it in items]), dtype=torch.float32),
        "target": torch.tensor(np.stack([it["target"] for it in items]), dtype=torch.float32),
        "questions": [it["question"] for it in items],
    }


def eval_predictions(pred: torch.Tensor, target: torch.Tensor) -> dict:
    cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
    mae = (pred - target).abs().mean().item()
    return {"cosine": float(cos), "mae": float(mae)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--v0-ckpt", default="checkpoints/v0/ylm_matscibert_s0.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--out", default="results/interface_generalization/results.json")
    args = ap.parse_args()

    labels = pickle.load(open(args.labels, "rb"))
    v0_qs = json.load(open(args.questions, encoding="utf-8"))
    chosen_mols, raw_splits = build_dataset(labels, v0_qs)
    print("Split question counts:", {k: len(v) for k, v in raw_splits.items()}, flush=True)

    # Pre-extract 60-epoch QM9 Encoder features once
    ck = torch.load(args.v0_ckpt, map_location="cpu", weights_only=False)
    enc = Encoder()
    enc.load_state_dict({k[len("encoder."):]: v for k, v in ck["state_dict"].items() if k.startswith("encoder.")})
    enc.eval()

    M = len(chosen_mols)
    Z = torch.zeros(M, N_MAX, dtype=torch.long)
    pos = torch.zeros(M, N_MAX, 3)
    mask = torch.zeros(M, N_MAX, dtype=torch.bool)
    forces = torch.zeros(M, N_MAX, 3)
    force_known = torch.zeros(M, N_MAX, dtype=torch.bool)
    rd_mols = []
    for m_new, old_m in enumerate(chosen_mols):
        lab = labels[old_m]
        n = len(lab["numbers"])
        Z[m_new, :n] = torch.tensor(lab["numbers"], dtype=torch.long)
        pos[m_new, :n] = torch.tensor(lab["pos"], dtype=torch.float32)
        mask[m_new, :n] = True
        f_arr = np.asarray(lab["forces"], dtype=np.float32)
        fin = np.isfinite(f_arr).all(axis=-1)
        force_known[m_new, :n] = torch.from_numpy(fin)
        f_clean = np.where(fin[:, None], f_arr, 0.0)
        forces[m_new, :n] = torch.from_numpy(f_clean)
        rd_mols.append(connectivity_mol(lab["numbers"], lab["pos"]))

    with torch.no_grad():
        X_feat = []
        for s in range(0, M, 128):
            X_feat.append(enc(Z[s:s + 128], pos[s:s + 128], mask[s:s + 128]))
        X = torch.cat(X_feat, dim=0)

    all_q = sorted({it["question"] for items in raw_splits.values() for it in items})
    print(f"Encoding {len(all_q)} unique questions with MiniLM...", flush=True)
    t_pad, t_mask, sid, text_dim = encode_texts(all_q)
    splits = {k: pack_split(v, sid) for k, v in raw_splits.items()}

    # Pre-parse all test splits once for the hand-written rule parser
    eval_names = ["rigid_single", "ood_synonym_single", "ood_synonym_comp", "group_ood"]
    rule_parsed = {}
    for name in eval_names:
        sp = splits[name]
        pt, pa = [], []
        for q_str, m_idx in zip(sp["questions"], sp["mol"].tolist()):
            t_res, a_res, b_res = resolve_program(q_str, rd_mols[m_idx])
            t_id = {"force": 0, "diff": 1, "proj": 2}.get(t_res, -1)
            pt.append(t_id)
            pa.append([a_res, b_res])
        rule_parsed[name] = (torch.tensor(pt, dtype=torch.long), torch.tensor(pa, dtype=torch.long))

    per_seed = defaultdict(lambda: defaultdict(list))
    train_sp = splits["train"]
    val_sp = splits["val"]
    rms_scale = train_sp["target"].pow(2).mean().sqrt().item()

    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1) RuleParser + Equivariant force readout head (trained on exact atom forces of train molecules)
        force_head = o3.Linear(enc.irreps_out, "1x1o")
        opt_f = torch.optim.AdamW(force_head.parameters(), lr=3e-3, weight_decay=1e-5)
        train_mols = torch.unique(train_sp["mol"])
        for _ in range(120):
            perm_m = train_mols[torch.randperm(len(train_mols))]
            for s_m in range(0, len(perm_m), 64):
                mb = perm_m[s_m:s_m + 64]
                v_pred = force_head(X[mb]) * rms_scale
                kn = force_known[mb]
                if kn.any():
                    loss_f = (((v_pred[kn] - forces[mb][kn]) / rms_scale) ** 2).mean()
                    opt_f.zero_grad()
                    loss_f.backward()
                    opt_f.step()
        force_head.eval()
        with torch.no_grad():
            V_all = (force_head(X) * rms_scale) * mask[..., None]
            for name in eval_names:
                sp = splits[name]
                pt, pa = rule_parsed[name]
                m_idx = sp["mol"]
                vm = V_all[m_idx]
                pm = pos[m_idx]
                b_ar = torch.arange(len(m_idx))
                ia, ib = pa[:, 0].clamp(min=0), pa[:, 1].clamp(min=0)
                u = pm[b_ar, ib] - pm[b_ar, ia]
                u = u / (u.pow(2).sum(-1, keepdim=True) + 1e-8).sqrt()
                y_rule = torch.zeros_like(sp["target"])
                ok1 = (pt == 0) & (pa[:, 0] >= 0)
                ok2 = (pt == 1) & (pa[:, 0] >= 0) & (pa[:, 1] >= 0)
                ok3 = (pt == 2) & (pa[:, 0] >= 0) & (pa[:, 1] >= 0)
                y_rule[ok1] = vm[b_ar[ok1], ia[ok1]]
                y_rule[ok2] = vm[b_ar[ok2], ia[ok2]] - vm[b_ar[ok2], ib[ok2]]
                f_a3 = vm[b_ar[ok3], ia[ok3]]
                u3 = u[ok3]
                y_rule[ok3] = (f_a3 * u3).sum(-1, keepdim=True) * u3
                metrics = eval_predictions(y_rule, sp["target"])
                parse_rate = float((ok1 | ok2 | ok3).float().mean().item())
                metrics["parse_rate"] = parse_rate
                per_seed["rule_exec"][name].append(metrics)

        # 2) Train & eval learned interfaces: optA_cont, optBC_cont, group_prog
        for m_key, builder in [
            ("optA_cont", lambda: FrozenOptA(text_dim, enc.irreps_out)),
            ("optBC_cont", lambda: FrozenOptBCLite(text_dim, enc.irreps_out)),
            ("group_prog", lambda: FrozenGroupProg(text_dim, enc.irreps_out)),
        ]:
            torch.manual_seed(seed)
            model = builder()
            opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
            best_val = float("inf")
            best_sd = None
            n_tr = len(train_sp["mol"])
            bs = 64
            for _ in range(args.epochs):
                model.train()
                perm = torch.randperm(n_tr)
                for s in range(0, n_tr, bs):
                    idx = perm[s:s + bs]
                    m_i = train_sp["mol"][idx]
                    u_i = train_sp["uid"][idx]
                    if m_key == "group_prog":
                        y_unit, aux = model(X[m_i], pos[m_i], mask[m_i], t_pad[u_i], t_mask[u_i])
                        y_raw = y_unit * rms_scale
                        loss = (((y_raw - train_sp["target"][idx]) / rms_scale) ** 2).mean()
                        ce = torch.nn.functional.cross_entropy
                        t_gt = train_sp["task"][idx]
                        loss = loss + 0.5 * ce(aux["task_logits"], t_gt)
                        is_ptr1 = (t_gt == 0) | (t_gt == 1) | (t_gt == 2)
                        is_ptr2 = (t_gt == 1) | (t_gt == 2)
                        is_grp = t_gt == 3
                        if is_ptr1.any():
                            loss = loss + 0.5 * ce(aux["ptr_logits"][0][is_ptr1], train_sp["atoms"][idx][is_ptr1, 0])
                        if is_ptr2.any():
                            loss = loss + 0.5 * ce(aux["ptr_logits"][1][is_ptr2], train_sp["atoms"][idx][is_ptr2, 1])
                        if is_grp.any():
                            gm_gt = train_sp["group_mask"][idx][is_grp]
                            am_sub = mask[m_i][is_grp]
                            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                                aux["group_logits"][is_grp], gm_gt, reduction="none"
                            )
                            loss = loss + 0.5 * (bce * am_sub).sum() / am_sub.sum().clamp(min=1)
                        # Auxiliary per-atom force supervision on known xTB atoms
                        v_cur = model.inner.last_V * rms_scale
                        kn = force_known[m_i]
                        if kn.any():
                            loss = loss + 0.5 * (((v_cur[kn] - forces[m_i][kn]) / rms_scale) ** 2).mean()
                    else:
                        y_raw = model(X[m_i], pos[m_i], mask[m_i], t_pad[u_i], t_mask[u_i]) * rms_scale
                        loss = (((y_raw - train_sp["target"][idx]) / rms_scale) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()

                model.eval()
                with torch.no_grad():
                    vm_i, vu_i = val_sp["mol"], val_sp["uid"]
                    vy = model(X[vm_i], pos[vm_i], mask[vm_i], t_pad[vu_i], t_mask[vu_i])
                    if m_key == "group_prog":
                        vy = vy[0] * rms_scale
                    else:
                        vy = vy * rms_scale
                    v_mae = (vy - val_sp["target"]).abs().mean().item()
                    if v_mae < best_val:
                        best_val = v_mae
                        best_sd = {k: v.clone() for k, v in model.state_dict().items()}

            assert best_sd is not None
            model.load_state_dict(best_sd)
            model.eval()
            with torch.no_grad():
                for name in eval_names:
                    sp = splits[name]
                    em_i, eu_i = sp["mol"], sp["uid"]
                    ey = model(X[em_i], pos[em_i], mask[em_i], t_pad[eu_i], t_mask[eu_i])
                    if m_key == "group_prog":
                        ey = ey[0] * rms_scale
                    else:
                        ey = ey * rms_scale
                    per_seed[m_key][name].append(eval_predictions(ey, sp["target"]))

        print(f"Finished seed {seed}", flush=True)

    # Verify full end-to-end O(3) rotation + reflection equivariance of GroupProgModel + Encoder
    from src.prog import GroupProgModel

    full_gp = GroupProgModel(text_dim=text_dim, d=128, key=64, encoder=enc).eval()
    full_gp.load_state_dict(model.inner.state_dict(), strict=False)
    sp0 = splits["group_ood"]
    sub_idx = torch.arange(min(32, len(sp0["mol"])))
    m0, u0 = sp0["mol"][sub_idx], sp0["uid"][sub_idx]
    rot_errs, ref_errs = [], []
    with torch.no_grad():
        y_ref, _ = full_gp(Z[m0], pos[m0], mask[m0], t_pad[u0], t_mask[u0])
        g = torch.Generator().manual_seed(0)
        for _ in range(4):
            R = o3.rand_matrix()
            yr, _ = full_gp(Z[m0], pos[m0] @ R.T, mask[m0], t_pad[u0], t_mask[u0])
            rot_errs.extend(((yr - y_ref @ R.T).norm(dim=-1) / y_ref.norm(dim=-1).clamp(min=1e-8)).tolist())
            Rm = -R
            ym, _ = full_gp(Z[m0], pos[m0] @ Rm.T, mask[m0], t_pad[u0], t_mask[u0])
            ref_errs.extend(((ym - y_ref @ Rm.T).norm(dim=-1) / y_ref.norm(dim=-1).clamp(min=1e-8)).tolist())

    summary = {
        "status": "MEASURED",
        "dataset": "Official TensorQA-v0.1.0 QM9 GFN2-xTB release (molecule-disjoint splits)",
        "encoder_backbone": args.v0_ckpt,
        "text_encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "seeds": args.seeds,
        "split_counts": {k: len(v) for k, v in raw_splits.items()},
        "group_prog_o3_equivariance": {
            "mean_rotation_rel_err": float(np.mean(rot_errs)),
            "mean_reflection_rel_err": float(np.mean(ref_errs)),
        },
        "interfaces": {},
    }
    for m_key, by_split in per_seed.items():
        summary["interfaces"][m_key] = {}
        for name, runs in by_split.items():
            cos_vals = [r["cosine"] for r in runs]
            mae_vals = [r["mae"] for r in runs]
            entry = {
                "cosine_mean": round(float(np.mean(cos_vals)), 4),
                "cosine_std": round(float(np.std(cos_vals)), 4),
                "mae_mean": round(float(np.mean(mae_vals)), 5),
                "mae_std": round(float(np.std(mae_vals)), 5),
            }
            if "parse_rate" in runs[0]:
                entry["parse_rate"] = round(float(runs[0]["parse_rate"]), 4)
            summary["interfaces"][m_key][name] = entry

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
