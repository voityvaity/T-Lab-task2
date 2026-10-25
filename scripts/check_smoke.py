"""Fail the manual CPU smoke if its multi-step controls did not really execute.

These are SOFTWARE integrity assertions, not evidence of recommender quality.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def validate_smoke_results(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("source_kind") != "synthetic" or data.get("status") != "synthetic_smoke_only":
        raise AssertionError("Smoke run must be prominently labeled synthetic")
    actual = data.get("test", {})
    required = {"gpt2", "rerank_0.0", "popularity", "sequential_rules", "smoothed_transition"}
    if not required.issubset(actual):
        raise AssertionError(f"Missing ranking controls: {required - set(actual)}")
    rl = [name for name in actual if name.startswith("rl_")]
    if not any("_h1_" in name for name in rl):
        raise AssertionError("No one-step RL model was tested")
    if not any("_h5_" in name for name in rl):
        raise AssertionError("No five-step RL model was tested")
    if not any("conservative" in name for name in rl):
        raise AssertionError("No uncertainty-aware RL variant was tested")
    for name, result in actual.items():
        for field in ("recall_at_10", "ndcg_at_10", "target_catalog_coverage"):
            number = result[field]
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise AssertionError(f"Invalid {name}.{field}: {number}")
        if result["queries"] <= 0:
            raise AssertionError(f"No offline evaluation for {name}")
    trajectories = data.get("real_five_step_windows", 0)
    if trajectories <= 0:
        raise AssertionError("Five-step drift requires genuinely consecutive held-out events")
    base_drift = data.get("real_context_vs_generated_context_drift", {}).get("gpt2", [])
    if len(base_drift) != 5 or not all(step["paired_trajectories"] > 0 for step in base_drift):
        raise AssertionError("Five-step drift evaluation silently skipped or empty")
    if data.get("selection_diagnostics", {}).get("selected") != data.get("selected_on_validation"):
        raise AssertionError("Selection report disagrees with selected checkpoint")
    return {"models_evaluated": len(actual), "rl_policies": len(rl),
            "real_five_step_windows": trajectories}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    result = validate_smoke_results(args.input)
    print("PASS: synthetic code-path smoke, NOT a scientific model-quality result:", result)


if __name__ == "__main__":
    main()
