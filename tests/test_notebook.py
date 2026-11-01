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
    for needle in ("OPTIONS_TO_RUN", "RUN_CPU_BENCH", "force_remount=True", "HEARTBEAT_SECONDS", "Popen"):
        assert needle in setup
