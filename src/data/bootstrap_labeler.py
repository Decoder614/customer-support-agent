"""Keyword-based bootstrap silver labeler for training and validation splits.

Generates reproducible silver annotations using high-precision domain regex heuristics
from src.intents.taxonomy. Tracks ambiguity flags, default escalation actions,
and outputs label distribution summaries.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml

from src.intents.taxonomy import (
    Intent,
    INTENTS,
    HIGH_RISK_INTENTS,
    propose,
    is_high_risk,
)


def determine_default_action(intent: str, is_ambiguous: bool) -> str:
    """Determines baseline action policy for bootstrapped data.

    Escalates if:
    - The intent is high-risk (safety, financial dispute, driver inquiry)
    - The utterance is ambiguous (0 or >1 keyword pattern matches)
    - The intent falls back to general feedback / unclassified
    """
    if is_high_risk(intent) or is_ambiguous or intent == Intent.OTHER_GENERAL_FEEDBACK.value:
        return "ESCALATE"
    return "AUTO_HANDLE"


def bootstrap_labels(
    df: pd.DataFrame,
    text_column: str = "text",
    fallback_column: str = "customer_message",
) -> pd.DataFrame:
    """Applies domain keyword heuristics to annotate a DataFrame with silver labels.

    Args:
        df: Input DataFrame containing customer utterances.
        text_column: Primary text column to inspect (e.g. normalized text).
        fallback_column: Fallback column if primary text is empty.

    Returns:
        DataFrame augmented with:
        - `intent`: Silver intent classification label
        - `ambiguity`: Boolean flag indicating keyword ambiguity
        - `action`: Baseline policy action ('ESCALATE' or 'AUTO_HANDLE')
        - `label_source`: Identifier constant 'keyword_bootstrap_NOT_HUMAN'
    """
    if df.empty:
        raise ValueError("Cannot bootstrap labels on an empty DataFrame.")

    df_out = df.copy()

    # Determine input text series
    if text_column in df_out.columns:
        text_series = df_out[text_column].fillna("")
    elif fallback_column in df_out.columns:
        text_series = df_out[fallback_column].fillna("")
    else:
        raise ValueError(f"Neither '{text_column}' nor '{fallback_column}' found in DataFrame.")

    # Apply taxonomy proposal heuristics
    proposals = text_series.map(propose)
    df_out["intent"] = proposals.map(lambda x: x[0])
    df_out["ambiguity"] = proposals.map(lambda x: bool(x[1]))
    df_out["label_source"] = "keyword_bootstrap_NOT_HUMAN"

    # Assign baseline policy action
    df_out["action"] = df_out.apply(
        lambda r: determine_default_action(r["intent"], r["ambiguity"]),
        axis=1,
    )

    return df_out


def generate_label_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes summary statistics on intent distributions and ambiguity."""
    total_rows = len(df)
    if total_rows == 0:
        return {"total_rows": 0}

    intent_counts = df["intent"].value_counts().to_dict()
    # Ensure all 9 intents are represented in summary
    for intent in INTENTS:
        intent_counts.setdefault(intent, 0)

    ambiguity_count = int(df["ambiguity"].sum()) if "ambiguity" in df.columns else 0
    action_counts = df["action"].value_counts().to_dict() if "action" in df.columns else {}

    summary = {
        "total_rows": total_rows,
        "intent_distribution": intent_counts,
        "ambiguity_count": ambiguity_count,
        "ambiguity_rate": round(ambiguity_count / max(1, total_rows), 4),
        "action_distribution": action_counts,
    }
    return summary


def format_summary_table(summary: Dict[str, Any], title: str = "Label Distribution Summary") -> str:
    """Formats distribution dictionary into a clean CLI table."""
    total = summary.get("total_rows", 0)
    lines = [
        f"\n{'='*70}",
        f"  {title} (N = {total:,})",
        f"{'='*70}",
        f"  {'Intent Category':<30} | {'Count':>8} | {'Pct (%)':>8} | {'Risk / Policy':<15}",
        f"  {'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*15}",
    ]

    intents_dist = summary.get("intent_distribution", {})
    for intent in INTENTS:
        cnt = intents_dist.get(intent, 0)
        pct = (cnt / max(1, total)) * 100
        risk_str = "ESCALATE (High)" if is_high_risk(intent) else "AUTO_HANDLE / Review"
        lines.append(f"  {intent:<30} | {cnt:>8} | {pct:>7.1f}% | {risk_str:<15}")

    lines.append(f"  {'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*15}")
    amb_cnt = summary.get("ambiguity_count", 0)
    amb_pct = summary.get("ambiguity_rate", 0) * 100
    lines.append(f"  Ambiguous / Multi-rule Flagged: {amb_cnt} ({amb_pct:.1f}%)")

    action_dist = summary.get("action_distribution", {})
    esc_cnt = action_dist.get("ESCALATE", 0)
    auto_cnt = action_dist.get("AUTO_HANDLE", 0)
    lines.append(f"  Action Breakdown: ESCALATE={esc_cnt} ({(esc_cnt/max(1, total))*100:.1f}%), AUTO_HANDLE={auto_cnt} ({(auto_cnt/max(1, total))*100:.1f}%)")
    lines.append(f"{'='*70}\n")
    return "\n".join(lines)


def process_splits(
    train_path: Path = Path("data/processed/train.csv"),
    dev_path: Optional[Path] = Path("data/processed/dev.csv"),
    eval_path: Optional[Path] = Path("data/processed/eval_bootstrap.csv"),
    artifacts_dir: Path = Path("artifacts"),
) -> Dict[str, Any]:
    """Bootstraps labels across train, dev, and eval splits, updating files and saving summary artifacts."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Train split not found at {train_path}. Run split creation first.")

    logging.info(f"Loading and bootstrapping silver labels for {train_path}...")
    train_df = pd.read_csv(train_path)
    train_df = bootstrap_labels(train_df)
    train_df.to_csv(train_path, index=False)

    all_dfs = [("train", train_df)]

    if dev_path and dev_path.exists():
        logging.info(f"Bootstrapping silver labels for {dev_path}...")
        dev_df = pd.read_csv(dev_path)
        dev_df = bootstrap_labels(dev_df)
        dev_df.to_csv(dev_path, index=False)
        all_dfs.append(("dev", dev_df))

    if eval_path and eval_path.exists():
        logging.info(f"Bootstrapping silver labels for {eval_path}...")
        eval_df = pd.read_csv(eval_path)
        eval_df = bootstrap_labels(eval_df)
        eval_df.to_csv(eval_path, index=False)
        all_dfs.append(("eval", eval_df))

    # Aggregate summaries
    train_summary = generate_label_summary(train_df)
    print(format_summary_table(train_summary, title="Train Split Silver Label Distribution"))

    # Save distribution artifacts
    combined_df = pd.concat([df.assign(split=split_name) for split_name, df in all_dfs], ignore_index=True)
    dist_table = combined_df.groupby(["split", "intent"]).size().unstack(fill_value=0)
    dist_table.to_csv(artifacts_dir / "intent_distribution.csv")

    # Save 5 representative examples per intent
    taxonomy_examples = {}
    for intent, grp in train_df.groupby("intent"):
        sample_cols = [c for c in ["example_id", "customer_message", "text"] if c in grp.columns]
        taxonomy_examples[intent] = grp.head(5)[sample_cols].to_dict(orient="records")

    with open(artifacts_dir / "taxonomy_examples.json", "w", encoding="utf-8") as f:
        json.dump(taxonomy_examples, f, indent=2)

    logging.info(f"Saved artifacts to {artifacts_dir / 'intent_distribution.csv'} and {artifacts_dir / 'taxonomy_examples.json'}")

    return train_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="data/processed/train.csv", help="Path to train CSV")
    parser.add_argument("--dev-path", default="data/processed/dev.csv", help="Path to dev CSV")
    parser.add_argument("--eval-path", default="data/processed/eval_bootstrap.csv", help="Path to eval bootstrap CSV")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for distribution artifacts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    process_splits(
        train_path=Path(args.train_path),
        dev_path=Path(args.dev_path) if args.dev_path else None,
        eval_path=Path(args.eval_path) if args.eval_path else None,
        artifacts_dir=Path(args.artifacts_dir),
    )


if __name__ == "__main__":
    main()
