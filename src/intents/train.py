"""Train and evaluate baseline and main candidate intent classifiers strictly on train.csv."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import yaml

from src.data.split import assert_no_leakage
from src.intents.taxonomy import Intent, INTENTS
from src.intents.baseline_classifiers import TrivialMajorityClassifier, SimpleTfidfClassifier
from src.intents.classifier import IntentClassifier


def compute_metrics(y_true: list, y_pred: list) -> Dict[str, float]:
    """Computes standard classification evaluation metrics."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }


def train_models(
    config_path: Path = Path("config/default.yaml"),
    train_path: Path = Path("data/processed/train.csv"),
    dev_path: Path = Path("data/processed/dev.csv"),
    eval_bootstrap_path: Path = Path("data/processed/eval_bootstrap.csv"),
    golden_path: Path = Path("data/golden/golden_eval_verified.csv"),
    models_dir: Path = Path("artifacts/models"),
    artifacts_dir: Path = Path("artifacts"),
) -> Dict[str, Any]:
    """Trains all 3 classification systems strictly on train.csv and validates on the golden benchmark.

    Returns:
        Summary metrics dictionary.
    """
    start_time = time.perf_counter()
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    cls_cfg = cfg.get("classification", {})
    c_val = float(cls_cfg.get("c_value", 4.0))
    word_ngrams = tuple(cls_cfg.get("word_ngram_range", [1, 2]))
    char_ngrams = tuple(cls_cfg.get("char_ngram_range", [3, 5]))

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {train_path}")

    logging.info(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path, dtype=str, keep_default_na=False)

    # Validate zero data leakage across all splits
    for split_name, path in [("Dev", dev_path), ("Eval Bootstrap", eval_bootstrap_path)]:
        if path.exists():
            other_df = pd.read_csv(path, dtype=str, keep_default_na=False)
            assert_no_leakage(train_df, other_df, split_names=("Train", split_name))

    train_texts = train_df["customer_message"].tolist()
    train_labels = train_df["intent"].tolist()

    # Initialize all 3 model tiers
    models = {
        "trivial": TrivialMajorityClassifier(),
        "simple": SimpleTfidfClassifier(c_value=1.0, random_state=seed),
        "main": IntentClassifier(
            c_value=c_val,
            word_ngram_range=word_ngrams,
            char_ngram_range=char_ngrams,
            class_weight="balanced",
            seed=seed,
        ),
    }

    # Fit and serialize each model
    logging.info("Fitting and saving model artifacts...")
    for name, model in models.items():
        model.fit(train_texts, train_labels)
        joblib.dump(model, models_dir / f"{name}.joblib")

    # Evaluate on Golden Benchmark if available
    benchmark_results = {}
    if golden_path.exists():
        gold_df = pd.read_csv(golden_path)
        gold_texts = gold_df["customer_message"].tolist()
        gold_labels = gold_df["intent_gold"].tolist()

        for name, model in models.items():
            preds = model.predict(gold_texts)
            metrics = compute_metrics(gold_labels, preds)
            benchmark_results[name] = metrics

    duration = time.perf_counter() - start_time
    train_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()

    metadata = {
        "training_rows": len(train_df),
        "training_sha256": train_sha,
        "seed": seed,
        "main_features": f"word {word_ngrams} + char_wb {char_ngrams} TF-IDF; balanced logistic regression C={c_val}",
        "simple_features": "word (1,1) TF-IDF; logistic regression C=1.0",
        "trivial_features": "majority class frequency",
        "golden_benchmark_results": benchmark_results,
        "runtime_seconds": round(duration, 3),
    }

    with open(artifacts_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Print summary table
    print(f"\n{'='*75}")
    print(f"  Model Training & Golden Evaluation Benchmark Summary (N_train = {len(train_df):,})")
    print(f"{'='*75}")
    print(f"  {'Model Tier':<20} | {'Golden Accuracy':>16} | {'Macro F1':>10} | {'Weighted F1':>12}")
    print(f"  {'-'*20}-+-{'-'*16}-+-{'-'*10}-+-{'-'*12}")
    for name, m in benchmark_results.items():
        print(f"  {name:<20} | {m['accuracy']*100:>15.1f}% | {m['macro_f1']:>10.3f} | {m['weighted_f1']:>12.3f}")
    print(f"{'='*75}\n")

    return metadata


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--train-path", default="data/processed/train.csv", help="Path to train CSV")
    parser.add_argument("--golden-path", default="data/golden/golden_eval_verified.csv", help="Golden benchmark CSV")
    parser.add_argument("--models-dir", default="artifacts/models", help="Directory for model joblibs")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    args = parser.parse_args()

    train_models(
        config_path=Path(args.config),
        train_path=Path(args.train_path),
        golden_path=Path(args.golden_path),
        models_dir=Path(args.models_dir),
        artifacts_dir=Path(args.artifacts_dir),
    )


if __name__ == "__main__":
    main()
