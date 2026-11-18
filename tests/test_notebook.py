"""Colab notebook hygiene: code cells compile, no Colab form annotations, settings present."""
import ast
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "colab" / "ylm_train.ipynb"


def _code_cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_code_cells_compile():
    for src in _code_cells():
        body = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("!", "%")))
        ast.parse(body)


def test_no_colab_form_fields():
    assert not any("@param" in src or "@title" in src for src in _code_cells())


def test_settings_and_mitigations_present():
    setup = _code_cells()[0]
    for needle in (
        "OPTIONS_TO_RUN",
        "RUN_CPU_BENCH",
        "force_remount=True",
        "HEARTBEAT_SECONDS",
        "Popen",
        "YLM_ATTN_SUP",
        "options_bc_test",
        "v0_multiseed",
    ):
        assert needle in setup
    assert "os.environ['YLM_OPTIONS']='B'" in setup
    assert "os.environ['YLM_ATTN_SUP']='0.1'" in setup
    assert "ATTN_SUP = float(os.environ.get('YLM_ATTN_SUP', '0.0'))" in setup


def _load_build_experiments():
    train_cell = _code_cells()[2]
    fragment = train_cell[: train_cell.index("\nexperiments = build_experiments()")]
    ns = {
        "MODE": "options",
        "FULL": True,
        "DATA_SOURCE": "v0_release",
        "OPTIONS_TO_RUN": "A,B,C",
        "TEXT_ENCODER": "minilm",
        "SEEDS": [0, 1, 2],
        "ATTN_SUP": 0.0,
        "INCLUDE_B_ATTN_SUP": False,
        "C_TP_MODE": "fully_connected",
        "FRACTIONS": "0.03,0.1,0.3",
        "FRACTION_TEXTS": "minilm,matscibert,taskid",
        "EVAL_ROTATIONS": 32,
        "EVAL_REFLECTIONS": False,
        "RUN_EXTRAS": False,
    }
    exec(fragment, ns)
    return ns["build_experiments"]


def test_attn_sup_only_on_option_b_and_distinct_run_name():
    build_experiments = _load_build_experiments()

    plain = build_experiments(seeds=[0, 1, 2], attn_sup=0.0)
    assert [(n, t, o) for n, t, o, _ in plain] == [
        ("v0_optA_minilm_s0", "minilm", "A"),
        ("v0_optB_minilm_s0", "minilm", "B"),
        ("v0_optC_minilm_s0", "minilm", "C"),
        ("v0_optA_minilm_s1", "minilm", "A"),
        ("v0_optB_minilm_s1", "minilm", "B"),
        ("v0_optC_minilm_s1", "minilm", "C"),
        ("v0_optA_minilm_s2", "minilm", "A"),
        ("v0_optB_minilm_s2", "minilm", "B"),
        ("v0_optC_minilm_s2", "minilm", "C"),
    ]
    assert all("--attn_sup" not in flags for *_, flags in plain)

    retry = build_experiments(seeds=[0], attn_sup=0.1, include_b_attn_sup=False)
    by_opt = {o: (n, flags) for n, _, o, flags in retry}
    assert by_opt["A"][0] == "v0_optA_minilm_s0" and "--attn_sup" not in by_opt["A"][1]
    assert by_opt["C"][0] == "v0_optC_minilm_s0" and "--attn_sup" not in by_opt["C"][1]
    assert by_opt["B"][0] == "v0_optB_attn0p1_minilm_s0" and "--attn_sup" in by_opt["B"][1]

    both_b = build_experiments(options_to_run="B", seeds=[0], attn_sup=0.1, include_b_attn_sup=True)
    assert [n for n, *_ in both_b] == ["v0_optB_minilm_s0", "v0_optB_attn0p1_minilm_s0"]

    smoke = build_experiments(options_to_run="B", seeds=[0], attn_sup=0.1, full=False, eval_rotations=4)
    assert smoke[0][0] == "tiny_optB_attn0p1_minilm_s0"
    assert "'--resume'" in _code_cells()[2]
