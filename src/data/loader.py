"""Kaggle dataset loader, brand filtering, and multi-turn thread reconstruction."""
import argparse
import logging
from pathlib import Path
import zipfile
from typing import Optional, Union

import pandas as pd
import yaml

from src.data.preprocessing import redact_pii, normalize_text

REQUIRED_COLUMNS = {
    'tweet_id', 'author_id', 'inbound', 'created_at',
    'text', 'response_tweet_id', 'in_response_to_tweet_id'
}


def load_raw_kaggle_subset(path: Union[str, Path] = 'data/raw/twcs.csv', max_rows: int = 150000) -> pd.DataFrame:
    """
    Load a bounded prefix of the Kaggle Customer Support on Twitter dataset.
    Supports .csv and .zip archives with automatic local directory discovery.
    """
    if max_rows <= 0:
        raise ValueError('max_rows must be greater than 0')
    
    path = Path(path)
    target_path = None

    # Check direct path first
    if path.exists():
        target_path = path
    else:
        # Check standard locations and alternatives
        search_candidates = [
            Path('data/raw') / path.name,
            Path('../data/raw') / path.name,
            Path('data/raw/customer-support-on-twitter.zip'),
            Path('../data/raw/customer-support-on-twitter.zip'),
            Path('data/raw/twcs.csv'),
            Path('../data/raw/twcs.csv'),
            Path('data/raw/twcs.zip'),
            Path('../data/raw/twcs.zip'),
        ]
        for candidate in search_candidates:
            if candidate.exists():
                target_path = candidate
                break

    # If still not found, scan data/raw directories for any .csv or .zip file
    if target_path is None:
        for raw_dir in [Path('data/raw'), Path('../data/raw')]:
            if raw_dir.exists():
                for f in raw_dir.iterdir():
                    if f.suffix.lower() in ['.csv', '.zip'] and not f.name.startswith('.'):
                        target_path = f
                        break
            if target_path is not None:
                break

    if target_path is None:
        raise FileNotFoundError(
            f"\n[!] Raw dataset file not found at '{path}'.\n"
            f"To fix this, please ensure the Kaggle 'Customer Support on Twitter' dataset is present:\n"
            f"1. Download 'Customer Support on Twitter' (twcs.csv or customer-support-on-twitter.zip) from Kaggle:\n"
            f"   https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter\n"
            f"2. Place either 'twcs.csv' or 'customer-support-on-twitter.zip' inside the 'data/raw/' directory:\n"
            f"   -> data/raw/twcs.csv  OR  data/raw/customer-support-on-twitter.zip\n"
        )

    logging.info(f"Loading raw dataset from {target_path} (max_rows={max_rows})...")
    path = target_path
    
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            csv_members = [m for m in archive.namelist() if m.endswith('.csv') and 'sample' not in m.lower()]
            if not csv_members:
                csv_members = [m for m in archive.namelist() if m.endswith('.csv')]
            if not csv_members:
                raise ValueError("No CSV file found inside the zip archive.")
            with archive.open(sorted(csv_members)[0]) as handle:
                df = pd.read_csv(handle, nrows=max_rows, dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(path, nrows=max_rows, dtype=str, keep_default_na=False)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")
    
    if not df.inbound.str.lower().isin(['true', 'false']).all():
        raise ValueError("Column 'inbound' must contain boolean strings ('true'/'false')")
    
    if df.tweet_id.eq('').any() or df.tweet_id.duplicated().any():
        raise ValueError("tweet_id must be non-empty and unique")
    
    df['inbound'] = df.inbound.str.lower().eq('true')
    return df


def extract_brand_dialogues(df: pd.DataFrame, brand: str = 'Uber_Support') -> pd.DataFrame:
    """
    Filter interactions for target brand and reconstruct conversation threads.
    Links customer queries with historical brand responses and ancestor context turns.
    """
    records = df.set_index('tweet_id').to_dict('index')
    parent = {key: key for key in records}

    # Union-Find / Disjoint Set Union (DSU) to group tweets into conversation components
    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for key, row in records.items():
        prev_id = row['in_response_to_tweet_id']
        if prev_id:
            root_a, root_b = find(key), find(prev_id)
            parent[max(root_a, root_b)] = min(root_a, root_b)

    results = []
    seen_pairs = set()

    # Sort reply IDs numerically for deterministic selection of historical replies
    sorted_tweet_ids = sorted(records.keys(), key=lambda x: (int(x) if x.isdigit() else 0, x))

    for reply_id in sorted_tweet_ids:
        reply_row = records[reply_id]
        
        # We look for outbound brand replies
        if reply_row['inbound'] or (brand and reply_row['author_id'] != brand):
            continue

        customer_id = reply_row['in_response_to_tweet_id']
        customer_row = records.get(customer_id)

        # Ensure preceding message was an inbound customer message
        if not customer_row or not customer_row['inbound']:
            continue

        if (customer_id, reply_row['author_id']) in seen_pairs:
            continue

        normalized_customer = normalize_text(customer_row['text'])
        normalized_reply = normalize_text(reply_row['text'])

        # Filter out trivial greetings or empty turns (< 8 characters)
        if len(normalized_customer) < 8 or not normalized_reply:
            continue

        seen_pairs.add((customer_id, reply_row['author_id']))

        # Reconstruct ancestor conversation context (up to 4 preceding turns)
        context_turns = []
        visited = {customer_id}
        ancestor_id = customer_row['in_response_to_tweet_id']

        while ancestor_id in records and ancestor_id not in visited and len(context_turns) < 4:
            visited.add(ancestor_id)
            ancestor = records[ancestor_id]
            role = 'customer' if ancestor['inbound'] else 'support'
            context_turns.append(f"{role}: {redact_pii(ancestor['text'])}")
            ancestor_id = ancestor['in_response_to_tweet_id']

        results.append({
            'example_id': customer_id,
            'reply_id': reply_id,
            'brand': reply_row['author_id'],
            'conversation_id': find(customer_id),
            'customer_message': redact_pii(customer_row['text']),
            'text': normalized_customer,
            'historical_brand_reply': redact_pii(reply_row['text']),
            'conversation_context': ' | '.join(reversed(context_turns)),
            'created_at': customer_row['created_at']
        })

    paired_df = pd.DataFrame(results)
    return paired_df


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # Load default configuration
    config_path = Path('config/default.yaml')
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    brand = cfg.get('brand', 'Uber_Support')
    data_cfg = cfg.get('data', {})
    max_rows = data_cfg.get('max_rows', 150000)
    raw_path = data_cfg.get('raw_path', 'data/raw/customer-support-on-twitter.zip')

    df_raw = load_raw_kaggle_subset(raw_path, max_rows=max_rows)
    logging.info(f"Loaded {len(df_raw)} tweets from raw dataset.")

    dialogues = extract_brand_dialogues(df_raw, brand=brand)
    logging.info(f"Extracted {len(dialogues)} paired interactions for brand '{brand}'.")

    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'paired_interactions.csv'
    dialogues.to_csv(output_path, index=False)
    logging.info(f"Saved paired interactions to {output_path} (unique conversations: {dialogues['conversation_id'].nunique() if not dialogues.empty else 0})")


if __name__ == '__main__':
    main()
