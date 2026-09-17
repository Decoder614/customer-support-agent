"""Tests for one-shot benchmark reproduction harness."""

import json
from pathlib import Path
import pytest

from scripts.reproduce_results import run_benchmark_reproduction


def test_reproduce_benchmark_execution(tmp_path: Path):
    """Verify that reproduction script runs end-to-end and produces structured artifacts."""
    artifacts_dir = tmp_path / "artifacts"
    models_dir = tmp_path / "models"

    summary = run_benchmark_reproduction(
        config_path=Path("config/default.yaml"),
        train_path=Path("data/processed/train.csv"),
        dev_path=Path("data/processed/dev.csv"),
        eval_bootstrap_path=Path("data/processed/eval_bootstrap.csv"),
        golden_path=Path("data/golden/golden_eval_verified.csv"),
        human_eval_path=Path("data/eval/judge_human_eval_completed.csv"),
        models_dir=models_dir,
        artifacts_dir=artifacts_dir,
        run_pytest=False,  # Skip nested pytest call inside unit test
    )

    assert "benchmark_results" in summary
    assert "trivial" in summary["benchmark_results"]
    assert "simple" in summary["benchmark_results"]
    assert "main" in summary["benchmark_results"]

    # Verify generated artifact files exist
    assert (artifacts_dir / "reproduction_summary.json").exists()
    assert (artifacts_dir / "benchmark_results.json").exists()
    assert (artifacts_dir / "threshold_sweep.json").exists()
    assert (artifacts_dir / "failure_cases.json").exists()
    assert (artifacts_dir / "judge_agreement.json").exists()

    # Check timing
    assert summary["timings_seconds"]["total_reproduction_seconds"] < 60.0

    # Check classification metric schema
    main_res = summary["benchmark_results"]["main"]
    assert main_res["intent_classification"]["accuracy"] > 0.60
    assert main_res["safety_metrics"]["auto_handle_rate"] > 0.0
