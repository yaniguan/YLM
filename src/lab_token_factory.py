"""Lab token factory: an agent proposes lab tasks, writes its reasoning, and only records
whose claims pass strict checkers are kept.

Two domains:
  protocol  The agent picks a goal (a recipe for one plate well), writes a protocol, and
            writes a step-by-step trace with its predicted verdict and volumes.
            Checkers: LabLean (Lean 4, YLM PR #92) gives the verdict; a Python copy of
            LabLean (src/lab_world.py) must agree with Lean; the claimed per-step volumes
            must equal the replayed ones.
  sql       The agent picks a question about a toy lab database, writes SQL, and states
            the answer. Checker: the 4-layer SQL gate (src/sql_gate.py) plus "the stated
            answer equals the executed result".

The "agent" here is a scripted program, not a language model. It makes mistakes on purpose:
its plans break rules (forgets to thaw, reuses tips, wrong pipette, bad temperature) and its
own reasoning has blind spots (it ignores some rules and sometimes slips on arithmetic). So
some of its claims are wrong, and the checkers have something real to catch.

Trusted regime: tasks have levels 1-4 (number of reagents in the recipe). A level opens only
after the level below it reaches the target verified yield in a round. The agent keeps a
memory of the claims that checkers refuted, and after each round it stops ignoring a rule
it was caught on (learning only from verified failures, no weight updates).

Every record carries provenance tags: who made it, with which seed, which checkers ran,
their versions, each verdict, and a trust label ("verified" or "rejected").
"""
import hashlib
import json
import random

from src import lab_world as W
from src import sql_gate as SG

FACTORY_VERSION = "lab-tf-v1"
GENERATOR = "scripted-agent-v1"
RULES = ("notThawed", "outOfRange", "contaminated", "overflow", "notEnough", "noTip", "badTemp")
REAGENTS = [("BUF", "buffer", "roomTemp", False), ("H2O", "water", "roomTemp", False),
            ("MM", "mastermix", "frozen", True), ("PRM", "primer", "frozen", True),
            ("TPL", "template", "frozen", True), ("ENZ", "enzyme", "onIce", True)]
WELLS = [f"A{i}" for i in range(1, 9)]
TUBE_MG, PLATE_MG = 1000, 14000      # plastic mass in mg, as in the LabLean examples


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ---------- the agent: propose a task and a protocol ----------

def propose_task(rng, level):
    """A world plus a goal: put `level` reagents into one well of plate P1."""
    stocks = rng.sample(REAGENTS, k=min(len(REAGENTS), level + 1))
    cs, rs = [], []
    for tid, name, state, cold in stocks:
        cs.append(W.container(tid, 15000, rng.choice([30, 80, 200, 1000, 5000]), name, tare=TUBE_MG))
        rs.append(W.reagent(name, state, cold, diluent=name == "water"))
    for w in WELLS:
        pre = rng.choice([0, 0, 0, 500, 1500, 1900])
        cs.append(W.container(w, 2000, pre, "water" if pre else None, plate="P1"))
        if pre and not any(r["name"] == "water" for r in rs):
            rs.append(W.reagent("water", diluent=True))
    target = rng.choice(WELLS)
    recipe = [(tid, rng.choice([10, 20, 25, 50, 100, 150, 200, 400])) for tid, *_ in stocks[:level]]
    return {"state": W.make_state(cs, rs, [("P1", PLATE_MG)]), "target": target, "recipe": recipe,
            "level": level}


def pick_pipette(v):
    return next(p for p, (lo, hi) in W.PIPETTES.items() if lo <= v <= hi)


def propose_protocol(rng, task, p_err=0.15):
    """The agent's plan. Each mistake kind is made with probability p_err."""
    s, steps, mistakes = task["state"], [], []
    frozen = {r["name"] for r in s["reagents"] if r["state"] == "frozen"}
    for tid, _v in task["recipe"]:
        name = W.find(s, tid)["contents"][0][0]
        if name in frozen:
            if rng.random() < p_err:
                mistakes.append("forgot thaw")
            else:
                steps.append(("thaw", name))
    holding = False
    for i, (tid, v) in enumerate(task["recipe"]):
        pip = pick_pipette(v)
        if rng.random() < p_err:
            pip = rng.choice([p for p in W.PIPETTES if p != pip])
            mistakes.append("wrong pipette")
        if holding and rng.random() < p_err:
            mistakes.append("reused tip")
        else:
            if holding:
                steps.append(("drop_tip",))
            if rng.random() < p_err / 2:
                mistakes.append("no tip")
            else:
                steps.append(("pick_tip",))
        holding = True
        steps.append(("transfer", pip, tid, task["target"], v))
    total = sum(v for _, v in task["recipe"])
    if total >= 100 and rng.random() < 0.7:
        mv = max(200, min(2000, total // 2 // 10 * 10))
        steps.append(("mix", "P200", task["target"], mv, 3))
    if holding:
        steps.append(("drop_tip",))
    steps.append(("seal", "P1"))
    temp = 37
    if rng.random() < p_err / 2:
        temp = rng.choice([2, 100, 110])
        mistakes.append("bad temperature")
    steps.append(("incubate", "P1", temp, 300))
    return steps, mistakes


# ---------- the agent: its own reasoning (may be wrong) ----------

def agent_trace(task, steps, rng, ignore, p_slip=0.15):
    """What the agent believes happens. Errors whose kind is in `ignore` are not noticed."""
    s, out = task["state"], []
    for st in steps:
        e = W.first_error(s, st)
        if e is not None and e[0] not in ignore:
            out.append((st, W.err_tag(e), {}))
            break
        before = {c["id"]: c["vol"] for c in s["cs"]}
        s = _apply_loose(s, st)
        out.append((st, None, {c["id"]: c["vol"] for c in s["cs"] if c["vol"] != before[c["id"]]}))
    if rng.random() < p_slip:                       # arithmetic slip in one claimed volume
        changed = [i for i, (_, e, ch) in enumerate(out) if ch]
        if changed:
            i = rng.choice(changed)
            st, e, ch = out[i]
            k = rng.choice(sorted(ch))
            out[i] = (st, e, {**ch, k: ch[k] + rng.choice([-50, -10, 10, 50])})
    return out


def _apply_loose(s, st):
    """apply(), but survives steps that break rules (the agent did not notice)."""
    if st[0] == "transfer":
        a = W.find(s, st[2])
        if a is None or W.find(s, st[3]) is None or a["vol"] == 0:
            return s
    if st[0] == "mix" and W.find(s, st[2]) is None:
        return s
    return W.apply(s, st)


def claim_of(tr):
    last = tr[-1]
    if last[1] is not None:
        return {"ok": False, "fail_step": len(tr), "error": last[1]}
    return {"ok": True, "fail_step": None, "error": None}


# ---------- text ----------

def start_text(task):
    s = task["state"]
    parts = [f"{c['id']} {W.fmt_vol(c['vol'])}" + (f" {c['contents'][0][0]}" if c["contents"] else "")
             for c in s["cs"] if c["plate"] is None or c["vol"] or c["id"] == task["target"]]
    frozen = [r["name"] for r in s["reagents"] if r["state"] == "frozen"]
    return "; ".join(parts) + (f"; frozen {' '.join(frozen)}" if frozen else "")


def goal_text(task):
    return f"fill {task['target']} with " + " + ".join(f"{W.fmt_vol(v)} {t}" for t, v in task["recipe"])


def record_text(task, steps, tr):
    """One record: goal, start, every step with its outcome and changed volumes, verdict."""
    out = [f"GOAL {goal_text(task)}", f"START {start_text(task)}"]
    for i, (st, err, ch) in enumerate(tr, 1):
        if err:
            out.append(f"{i} {W.step_text(st)}: FAIL {err}")
        else:
            vols = " ".join(f"{k}={W.fmt_vol(v)}" for k, v in sorted(ch.items()))
            out.append(f"{i} {W.step_text(st)}: ok" + (f" {vols}" if vols else ""))
    c = claim_of(tr)
    out.append("VERDICT pass" if c["ok"] else f"VERDICT fail at step {c['fail_step']}: {c['error']}")
    return " | ".join(out)


def sql_record_text(question, sql, answer):
    return f"QUESTION {question} | SQL {sql} | ANSWER {answer}"


# ---------- the factory loop ----------

class Agent:
    """Blind spots plus a memory of refuted claims (learning only from verified failures)."""

    def __init__(self, rng, p_blind=0.5, learn=True):
        self.ignore = {r for r in RULES if rng.random() < p_blind}
        self.learn = learn
        self.refuted = {}

    def update(self, refuted_rules, min_count=3):
        for r in refuted_rules:
            self.refuted[r] = self.refuted.get(r, 0) + 1
        if self.learn:
            self.ignore -= {r for r, n in self.refuted.items() if n >= min_count}


def make_protocol_records(rng, n, level, agent, p_err=0.15, p_slip=0.15):
    recs = []
    for _ in range(n):
        task = propose_task(rng, level)
        steps, mistakes = propose_protocol(rng, task, p_err)
        tr = agent_trace(task, steps, rng, agent.ignore, p_slip)
        recs.append({"task": task, "steps": steps, "trace": tr, "mistakes": mistakes,
                     "text": record_text(task, steps, tr)})
    return recs


def verify_protocols(recs, lablean_dir=None):
    """Run the checkers in order; fill rec['verdicts'] and rec['trust']. Lean runs in one batch."""
    use_lean = W.lean_available(lablean_dir)
    lean = W.lean_check([(r["task"]["state"], r["steps"]) for r in recs], lablean_dir) if use_lean \
        else [None] * len(recs)
    for r, lv in zip(recs, lean):
        v = {}
        mirror = W.check(r["task"]["state"], r["steps"])
        truth = lv if use_lean else mirror
        claim = claim_of(r["trace"])
        v["V0_format"] = "pass" if r["text"].isascii() and "\n" not in r["text"] else "fail"
        if use_lean:
            v["V1_lablean"] = "pass" if (claim["ok"], claim["fail_step"], claim["error"]) == \
                (lv["ok"], lv["fail_step"], lv["error"]) else "fail"
            v["V2_mirror_agrees"] = "pass" if mirror == lv else "fail"
        else:
            v["V1_lablean"] = "skipped"
            v["V2_mirror_agrees"] = "skipped"
            v["V1b_mirror"] = "pass" if (claim["ok"], claim["fail_step"], claim["error"]) == \
                (mirror["ok"], mirror["fail_step"], mirror["error"]) else "fail"
        true_tr = W.trace(r["task"]["state"], r["steps"])
        v["V3_volumes"] = "pass" if [x[2] for x in true_tr] == [x[2] for x in r["trace"]] else "fail"
        r["verdicts"] = v
        r["truth"] = truth
        r["first_fail"] = next((k for k, x in v.items() if x == "fail"), None)
        r["trust"] = "verified" if r["first_fail"] is None else "rejected"
        # which rule the agent missed (claimed less than the truth), for its memory
        r["refuted_rule"] = None
        if r["trust"] == "rejected" and not truth["ok"] and \
                (claim["ok"] or claim["fail_step"] > truth["fail_step"]):
            r["refuted_rule"] = truth["error"].split()[0]
    return use_lean


def make_sql_records(rng, n, con_seed, p_wrong=0.3, p_wrong_answer=0.1):
    con = SG.build_db(con_seed)
    second = SG.build_db(con_seed + 7919)
    qs = SG.questions()
    recs = []
    for _ in range(n):
        q, gold, wrongs = rng.choice(qs)
        label, sql = ("correct", gold)
        if rng.random() < p_wrong:
            label, sql = rng.choice(wrongs)
        try:
            res = SG.run(con, sql) if label != "not read-only" else None
        except Exception:                                    # the agent could not run it
            res = None
        answer = "unknown" if not res else ", ".join(str(x) for row in sorted(res, key=str) for x in row)
        if rng.random() < p_wrong_answer and answer != "unknown":
            answer = answer + "0"                             # a copying slip in the stated answer
        recs.append({"question": q, "sql": sql, "label": label, "answer": answer, "gold": gold,
                     "con_seed": con_seed, "text": sql_record_text(q, sql, answer)})
    for r in recs:
        layer, reason, got = SG.gate(con, r["sql"], r["gold"], [second])
        v = {f"S{i + 1}_{name}": ("pass" if layer is None or i < layer else ("fail" if i == layer else "not run"))
             for i, name in enumerate(SG.LAYERS)}
        executed = None if got is None else ", ".join(str(x) for row in sorted(got, key=str) for x in row)
        v["S5_answer"] = "pass" if layer is None and executed == r["answer"] else \
            ("not run" if layer is not None else "fail")
        r["verdicts"], r["gate_reason"] = v, reason
        r["first_fail"] = next((k for k, x in v.items() if x == "fail"), None)
        r["trust"] = "verified" if r["first_fail"] is None else "rejected"
    return recs


def provenance(rec, domain, seed, round_i, checker_versions):
    return {"id": sha(rec["text"])[:16], "text_sha256": sha(rec["text"]), "source": "agent",
            "generator": GENERATOR, "factory": FACTORY_VERSION, "domain": domain, "seed": seed,
            "round": round_i, "level": rec.get("task", {}).get("level"),
            "checkers": checker_versions, "verdicts": rec["verdicts"], "trust": rec["trust"],
            "data": "synthetic"}


def run_factory(seed=0, rounds=6, per_round=200, sql_per_round=50, lablean_dir=None,
                open_at=0.7, p_blind=0.5, learn=True, checker_versions=None, seen=None):
    """The gated loop. Returns (kept, rejected, stats).

    `seen` is the set of kept text hashes; pass the same set to every seed so that a record
    already kept by an earlier seed is rejected as "dedup" (it is updated in place)."""
    rng = random.Random(seed)
    agent = Agent(rng, p_blind, learn)
    frontier = 1
    seen = set() if seen is None else seen
    kept, rejected, rounds_log = [], [], []
    used_lean = None
    for k in range(rounds):
        recs = []
        for level in range(1, frontier + 1):
            recs += make_protocol_records(rng, per_round // frontier, level, agent)
        used_lean = verify_protocols(recs, lablean_dir)
        sql = make_sql_records(rng, sql_per_round, con_seed=seed * 1000 + k)
        by_level = {}
        for r in recs + sql:
            dom = "protocol" if "task" in r else "sql"
            if r["trust"] == "verified" and sha(r["text"]) in seen:
                r["trust"], r["first_fail"] = "rejected", "dedup"
            p = provenance(r, dom, seed, k, checker_versions or {})
            p["trust"], p["first_fail"] = r["trust"], r["first_fail"]
            row = {"text": r["text"], "provenance": p}
            if dom == "sql":
                row["label"] = r["label"]
            else:
                row["mistakes"] = r["mistakes"]
                by_level.setdefault(r["task"]["level"], []).append(r["trust"] == "verified")
            if r["trust"] == "verified":
                seen.add(sha(r["text"]))
                kept.append(row)
            else:
                rejected.append(row)
        yields = {lv: sum(x) / len(x) for lv, x in sorted(by_level.items())}
        refuted = [r["refuted_rule"] for r in recs if r.get("refuted_rule")]
        log = {"round": k, "frontier": frontier, "ignore_before": sorted(agent.ignore),
               "protocol_yield_by_level": yields,
               "protocol_kept": sum(r["trust"] == "verified" for r in recs), "protocol_total": len(recs),
               "sql_kept": sum(r["trust"] == "verified" for r in sql), "sql_total": len(sql),
               "sql_wrong_but_passed": sum(r["trust"] == "verified" and r["label"] != "correct" for r in sql)}
        agent.update(refuted)
        if yields.get(frontier, 0) >= open_at and frontier < 4:
            frontier += 1
        log["frontier_after"] = frontier
        rounds_log.append(log)
    return kept, rejected, {"rounds": rounds_log, "used_lean": used_lean, "seed": seed}
