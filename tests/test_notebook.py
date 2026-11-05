"""Colab notebook hygiene: code cells compile, no Colab form annotations, settings present."""
import ast
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "colab" / "ylm_train.ipynb"


def _code_cells():
    nb = json.loads(NB.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_code_cells_compile():
    for src in _code_cells():
        body = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        ast.parse(body)


def test_no_colab_form_fields():
    assert not any("@param" in src or "@title" in src for src in _code_cells())


def test_settings_and_mitigations_present():
    setup = _code_cells()[0]
    for needle in ("OPTIONS_TO_RUN", "RUN_CPU_BENCH", "force_remount=True", "HEARTBEAT_SECONDS", "Popen",
                   "YLM_ATTN_SUP", "options_bc_test"):
        assert needle in setup
    assert "os.environ['YLM_OPTIONS']='B'" in setup
    assert "os.environ['YLM_ATTN_SUP']='0.1'" in setup
    assert "ATTN_SUP = float(os.environ.get('YLM_ATTN_SUP', '0.0'))" in setup


def _experiments(options, attn_sup, full=True):
    """Run the notebook's experiment-list fragment with stubbed settings."""
    train = _code_cells()[2]
    fragment = train[train.index("wanted = "):train.index("if RUN_EXTRAS:")]
    ns = dict(OPTIONS_TO_RUN=options, ATTN_SUP=attn_sup, P="full_" if full else "tiny_")
    exec(fragment, ns)
    return ns["experiments"]


def test_attn_sup_only_on_option_b_and_distinct_run_name():
    plain = _experiments("A,B,C", 0.0)
    assert [(n, t, o, e) for n, t, o, e in plain] == [
        ("full_A", "minilm", "A", []),
        ("full_B", "minilm", "B", []),
        ("full_C", "minilm", "C", []),
    ]
    retry = _experiments("A,B,C", 0.1)
    by_option = {o: (n, e) for n, _, o, e in retry}
    assert by_option["A"] == ("full_A", [])
    assert by_option["C"] == ("full_C", [])
    assert by_option["B"] == ("full_B_attn0p1", ["--attn_sup", "0.1"])
    assert _experiments("B", 0.1, full=False) == [
        ("tiny_B_attn0p1", "minilm", "B", ["--attn_sup", "0.1"])
    ]
    assert "'--resume'" in _code_cells()[2]
