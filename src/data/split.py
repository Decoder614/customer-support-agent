"""Group-aware and near-duplicate-aware leak-free dataset partitioning."""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import yaml


def assert_no_leakage(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    columns: Tuple[str, ...] = ('example_id', 'conversation_id', 'split_group', 'text'),
    split_names: Tuple[str, str] = ('Split A', 'Split B')
) -> None:
    """
    Assert that there is zero overlap between two dataset splits across key identifiers
    and near-duplicate group hashes.
    """
    for col in columns:
        if col in df_a.columns and col in df_b.columns:
            overlap = set(df_a[col].astype(str)) & set(df_b[col].astype(str))
            # Exclude empty string matches
            overlap.discard('')
            if overlap:
                sample_leaks = list(overlap)[:5]
                raise ValueError(
                    f"Data leakage detected between {split_names[0]} and {split_names[1]} on '{col}': "
                    f"{len(overlap)} shared values (e.g. {sample_leaks})"
                )


def create_leak_free_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    eval_ratio: float = 0.15,
    seed: int = 42,
    near_dup_radius: float = 0.10
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition interactions into train, dev, and eval sets while preventing data leakage.
    
    Ensures:
    1. All turns belonging to the same conversation thread stay in the same split.
    2. Near-duplicate customer utterances (cosine distance <= near_dup_radius) stay in the same split.
    """
    if df.empty:
        raise ValueError("Cannot partition an empty DataFrame.")
    
    df = df.reset_index(drop=True).copy()
    n_records = len(df)
    parent = list(range(n_records))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    # 1. Union records that share the same conversation_id
    if 'conversation_id' in df.columns:
        for indices in df.groupby('conversation_id').indices.values():
            for idx in indices[1:]:
                union(int(indices[0]), int(idx))

    # 2. Union near-duplicate text utterances using character n-gram cosine similarity
    text_series = df['text'].fillna('')
    if (text_series.str.len() > 0).any():
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1)
        features = vectorizer.fit_transform(text_series)
        
        # Radius neighbors: distance <= near_dup_radius -> cosine similarity >= 1 - radius
        nn = NearestNeighbors(metric='cosine', radius=near_dup_radius, algorithm='brute')
        nn.fit(features)
        
        for i, neighbor_indices in enumerate(nn.radius_neighbors(features, return_distance=False)):
            for j in neighbor_indices:
                union(i, int(j))

    # 3. Assign split_group based on disjoint sets
    df['split_group'] = [str(find(i)) for i in range(n_records)]
    unique_groups = sorted(df['split_group'].unique())

    # 4. Deterministic pseudo-random shuffle of groups
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    train_cutoff = int(train_ratio * n_groups)
    dev_cutoff = int((train_ratio + dev_ratio) * n_groups)

    group_to_split: Dict[str, str] = {}
    for idx, grp in enumerate(unique_groups):
        if idx < train_cutoff:
            group_to_split[grp] = 'train'
        elif idx < dev_cutoff:
            group_to_split[grp] = 'dev'
        else:
            group_to_split[grp] = 'eval'

    df['split'] = df['split_group'].map(group_to_split)

    train_df = df[df['split'] == 'train'].copy().reset_index(drop=True)
    dev_df = df[df['split'] == 'dev'].copy().reset_index(drop=True)
    eval_df = df[df['split'] == 'eval'].copy().reset_index(drop=True)

    # 5. Assert 0 leakage between all split pairs
    assert_no_leakage(train_df, dev_df, split_names=('Train', 'Dev'))
    assert_no_leakage(train_df, eval_df, split_names=('Train', 'Eval'))
    assert_no_leakage(dev_df, eval_df, split_names=('Dev', 'Eval'))

    return train_df, dev_df, eval_df


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    config_path = Path('config/default.yaml')
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    data_cfg = cfg.get('data', {})
    seed = cfg.get('seed', 42)
    train_ratio = data_cfg.get('train_ratio', 0.70)
    dev_ratio = data_cfg.get('dev_ratio', 0.15)
    eval_ratio = data_cfg.get('eval_ratio', 0.15)

    input_path = Path('data/processed/paired_interactions.csv')
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at {input_path}. Please run src.data.loader first.")

    logging.info(f"Loading paired interactions from {input_path}...")
    df = pd.read_csv(input_path)

    logging.info(f"Creating leak-free splits with seed={seed} (train={train_ratio}, dev={dev_ratio}, eval={eval_ratio})...")
    train_df, dev_df, eval_df = create_leak_free_splits(
        df,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        eval_ratio=eval_ratio,
        seed=seed
    )

    output_dir = Path('data/processed')
    train_df.to_csv(output_dir / 'train.csv', index=False)
    dev_df.to_csv(output_dir / 'dev.csv', index=False)
    eval_df.to_csv(output_dir / 'eval_bootstrap.csv', index=False)

    logging.info(f"Saved splits successfully:")
    logging.info(f"  - Train: {len(train_df)} rows ({train_df['conversation_id'].nunique()} conversations)")
    logging.info(f"  - Dev:   {len(dev_df)} rows ({dev_df['conversation_id'].nunique()} conversations)")
    logging.info(f"  - Eval:  {len(eval_df)} rows ({eval_df['conversation_id'].nunique()} conversations)")


if __name__ == '__main__':
    main()
