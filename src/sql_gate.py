"""A strict, layered gate for SQL that the lab agent writes over a toy lab database.

Same four layers as the SQL gate in github.com/yaniguan/lab-verify (demo/sql_gate.py), but
built only on the standard library (sqlite3), so YLM needs no new dependency:
  1 parse      sqlite can compile the text
  2 read-only  one statement (checked before sqlite sees it), and the sqlite authorizer sees only SELECT / READ / FUNCTION
  3 schema     every table and column exists (sqlite reports "no such table/column")
  4 execution  the result multiset equals the gold query's result on the main database
               AND on a second database with other rows (a wrong query can match gold
               on one database by luck, e.g. both return NULL)
The database rows are SYNTHETIC (made from a seed), not real lab records.
"""
import random
import sqlite3
from collections import Counter

SQL_GATE_VERSION = "sql-gate-v1"
SCHEMA = {
    "experiments": ["id", "operator", "protocol", "temperature_c", "yield_pct", "status"],
    "samples": ["id", "experiment_id", "well", "reagent", "volume_ul"],
}
LAYERS = ("parse", "read-only", "schema", "execution")
_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}


def build_db(seed=0, n_exp=30):
    rng = random.Random(seed)
    con = sqlite3.connect(":memory:")
    for t, cols in SCHEMA.items():
        con.execute(f"CREATE TABLE {t} ({', '.join(cols)})")
    sid = 100
    for i in range(1, n_exp + 1):
        con.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)",
                    (i, rng.choice(["Kim", "Patel", "Novak", "Diaz"]),
                     rng.choice(["PCR", "ELISA", "Western", "Miniprep"]),
                     rng.choice([4, 25, 37, 42, 65, 95]), round(rng.uniform(20, 95), 1),
                     rng.choice(["done", "done", "failed", "running"])))
        for _ in range(rng.randint(0, 3)):
            sid += 1
            con.execute("INSERT INTO samples VALUES (?,?,?,?,?)",
                        (sid, i, f"{'ABCDEFGH'[rng.randint(0, 7)]}{rng.randint(1, 12)}",
                         rng.choice(["Taq", "buffer", "dNTP", "primer"]),
                         rng.choice([2.0, 5.0, 10.0, 20.0, 25.0])))
    con.commit()
    return con


def run(con, sql):
    rows = con.execute(sql).fetchall()
    return Counter(tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in rows)


def more_than_one_statement(sql):
    """True if text follows the first complete statement (a ';' outside quotes and comments)."""
    for i, ch in enumerate(sql):
        if ch == ";" and sqlite3.complete_statement(sql[:i + 1]):
            return sql[i + 1:].strip() != ""
    return False


def gate(con, sql, gold_sql, extra_cons=()):
    """Return (index of first failed layer or None, reason, result Counter on `con` or None)."""
    if not sqlite3.complete_statement(sql.strip().rstrip(";") + ";"):
        return 0, "incomplete statement", None
    if more_than_one_statement(sql):
        return 1, "more than one statement", None
    bad = []

    def auth(action, *_):
        if action in _ALLOWED:
            return sqlite3.SQLITE_OK
        bad.append(action)
        return sqlite3.SQLITE_DENY

    con.set_authorizer(auth)
    try:
        con.execute("EXPLAIN " + sql)
    except (sqlite3.ProgrammingError, sqlite3.Warning) as e:   # more than one statement
        return 1, str(e), None                                # (Python 3.10/3.11 raise Warning)
    except sqlite3.DatabaseError as e:
        msg = str(e)
        if bad or "not authorized" in msg:
            return 1, "not read-only", None
        if "no such table" in msg or "no such column" in msg:
            return 2, msg, None
        return 0, msg, None
    finally:
        con.set_authorizer(None)
    if bad:
        return 1, "not read-only", None
    con.set_authorizer(auth)
    try:
        got = run(con, sql)
    finally:
        con.set_authorizer(None)
    gold = run(con, gold_sql)
    if got != gold:
        return 3, f"{sum(got.values())} rows vs {sum(gold.values())} gold rows" if \
            sum(got.values()) != 1 or sum(gold.values()) != 1 else \
            f"value {list(got)[0]} vs gold {list(gold)[0]}", got
    for other in extra_cons:
        if run(other, sql) != run(other, gold_sql):
            return 3, "differs from gold on a second database", got
    return None, "match", got


def lucky_pass_counts(seeds):
    """How many wrong candidate queries pass the gate with one database, and with two.

    Every wrong variant of every question is gated on build_db(seed), and again with
    build_db(seed + 7919) as the second database (the same pairing the factory uses)."""
    wrong = one_db = two_db = 0
    for seed in seeds:
        con, other = build_db(seed), build_db(seed + 7919)
        for _q, gold, wrongs in questions():
            for _label, sql in wrongs:
                wrong += 1
                if gate(con, sql, gold)[0] is None:
                    one_db += 1
                    if gate(con, sql, gold, [other])[0] is None:
                        two_db += 1
    return {"wrong_queries": wrong, "pass_one_db": one_db, "pass_two_db": two_db}


# ---------- questions with gold SQL (the agent picks one and writes its own SQL) ----------

def questions():
    """(question, gold sql, list of typical wrong variants (label, sql))."""
    out = []
    for p in ["PCR", "ELISA", "Western", "Miniprep"]:
        out.append((f"How many {p} experiments are done?",
                    f"SELECT COUNT(*) FROM experiments WHERE protocol = '{p}' AND status = 'done'",
                    [("missing filter", f"SELECT COUNT(*) FROM experiments WHERE protocol = '{p}'"),
                     ("wrong column", f"SELECT COUNT(*) FROM experiments WHERE assay = '{p}' AND status = 'done'")]))
        out.append((f"What is the mean yield of {p} experiments?",
                    f"SELECT ROUND(AVG(yield_pct), 1) FROM experiments WHERE protocol = '{p}'",
                    [("MAX vs AVG", f"SELECT ROUND(MAX(yield_pct), 1) FROM experiments WHERE protocol = '{p}'"),
                     ("not read-only", f"DELETE FROM experiments WHERE protocol = '{p}'")]))
    for r in ["Taq", "buffer", "dNTP", "primer"]:
        out.append((f"How many experiments used {r}?",
                    f"SELECT COUNT(DISTINCT experiment_id) FROM samples WHERE reagent = '{r}'",
                    [("COUNT vs COUNT DISTINCT", f"SELECT COUNT(experiment_id) FROM samples WHERE reagent = '{r}'"),
                     ("syntax error", f"SELECT COUNT(DISTINCT experiment_id FROM samples WHERE reagent = '{r}'")]))
        out.append((f"What total volume of {r} was used in failed experiments?",
                    "SELECT ROUND(SUM(s.volume_ul), 1) FROM samples s JOIN experiments e "
                    f"ON s.experiment_id = e.id WHERE s.reagent = '{r}' AND e.status = 'failed'",
                    [("wrong join", "SELECT ROUND(SUM(s.volume_ul), 1) FROM samples s JOIN experiments e "
                      f"ON s.id = e.id WHERE s.reagent = '{r}' AND e.status = 'failed'"),
                     ("stacked statement", f"SELECT 1; DROP TABLE samples")]))
    return out
