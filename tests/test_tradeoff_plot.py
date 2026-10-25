"""SVG figures must be valid XML and must label synthetic runs honestly."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET


def renderer():
    source = Path(__file__).resolve().parents[1] / "scripts" / "plot_tradeoff.py"
    spec = importlib.util.spec_from_file_location("tlab2_plot", source)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod.draw_tradeoff


def test_tradeoff_svg_is_parseable_and_source_labeled(tmp_path: Path):
    file = tmp_path / "tradeoff.svg"
    draw = renderer()
    data = {
        "gpt2": {"ndcg_at_10": 0.02, "auditor_completion_at_10": .60},
        "rl_completion_h1_kl_0.1": {
            "ndcg_at_10": 0.0195, "auditor_completion_at_10": .63
        },
    }
    draw(data, source_kind="synthetic", split="validation", selected="gpt2",
         output=file)
    root = ET.parse(file).getroot()
    assert root.tag.endswith("svg")
    assert "SYNTHETIC SOFTWARE SMOKE" in file.read_text()
    assert "rl_completion" in file.read_text()


def test_tradeoff_svg_handles_equal_coordinates(tmp_path: Path):
    file = tmp_path / "scatter.svg"
    renderer()({"gpt2": {"ndcg_at_10": .2,
                           "auditor_completion_at_10": .7}},
               source_kind="yambda", split="test", selected="gpt2", output=file)
    assert ET.parse(file).getroot().tag.endswith("svg")
