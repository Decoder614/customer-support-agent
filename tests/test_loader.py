"""Unit tests for dataset loading, brand filtering, and dialogue reconstruction."""
import pandas as pd
import pytest
from src.data.loader import extract_brand_dialogues, load_raw_kaggle_subset


def test_extract_brand_dialogues_basic():
    # Construct a synthetic multi-turn tweet dataframe
    data = {
        'tweet_id': ['1', '2', '3', '4'],
        'author_id': ['cust_1', 'Uber_Support', 'cust_1', 'Uber_Support'],
        'inbound': ['true', 'false', 'true', 'false'],
        'created_at': ['2026-01-01', '2026-01-01', '2026-01-01', '2026-01-01'],
        'text': [
            'I left my jacket in the car!',
            'Please DM us your email to help.',
            'DM sent with my details.',
            'Thanks, we received it!'
        ],
        'response_tweet_id': ['2', '', '4', ''],
        'in_response_to_tweet_id': ['', '1', '2', '3']
    }
    df = pd.DataFrame(data)
    df['inbound'] = df.inbound.str.lower().eq('true')

    dialogues = extract_brand_dialogues(df, brand='Uber_Support')
    assert len(dialogues) == 2
    assert set(dialogues.columns) >= {
        'example_id', 'reply_id', 'brand', 'conversation_id',
        'customer_message', 'text', 'historical_brand_reply', 'conversation_context'
    }
    # Check conversation thread grouping
    assert dialogues.iloc[0]['conversation_id'] == dialogues.iloc[1]['conversation_id']
    # Check context in turn 2
    assert 'customer: I left my jacket in the car!' in dialogues.iloc[1]['conversation_context']


def test_extract_brand_dialogues_filter_other_brand():
    data = {
        'tweet_id': ['1', '2'],
        'author_id': ['cust_1', 'AppleSupport'],
        'inbound': ['true', 'false'],
        'created_at': ['2026-01-01', '2026-01-01'],
        'text': ['My iPhone screen is broken', 'Please visit Apple Store'],
        'response_tweet_id': ['2', ''],
        'in_response_to_tweet_id': ['', '1']
    }
    df = pd.DataFrame(data)
    df['inbound'] = df.inbound.str.lower().eq('true')

    dialogues = extract_brand_dialogues(df, brand='Uber_Support')
    assert len(dialogues) == 0
