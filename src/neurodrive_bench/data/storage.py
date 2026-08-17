from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def save_dataset(records: list[dict[str, Any]], path: str | Path) -> None:
    """Saves records to a Parquet file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not records:
        print("Warning: No records to save.")
        return
        
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)
    print(f"Saved dataset with {len(records)} frames to {output_path}")


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Loads records from a Parquet file."""
    table = pq.read_table(path)
    return table.to_pylist()


def split_dataset(
    records: list[dict[str, Any]], 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.2, 
    seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Splits the dataset into train, validation, and test sets.
    Splitting is done by episode_id to prevent data leakage between overlapping windows.
    """
    assert abs((train_ratio + val_ratio) - 1.0) < 1e-5 or train_ratio + val_ratio <= 1.0
    
    # Group by episode
    episodes: dict[str, list[dict]] = {}
    for record in records:
        episodes.setdefault(str(record["episode_id"]), []).append(record)
        
    episode_ids = sorted(list(episodes.keys()))
    
    random.seed(seed)
    random.shuffle(episode_ids)
    
    num_episodes = len(episode_ids)
    num_train = int(num_episodes * train_ratio)
    num_val = int(num_episodes * val_ratio)
    
    train_ids = episode_ids[:num_train]
    val_ids = episode_ids[num_train:num_train + num_val]
    test_ids = episode_ids[num_train + num_val:]
    
    train_records = [record for ep_id in train_ids for record in episodes[ep_id]]
    val_records = [record for ep_id in val_ids for record in episodes[ep_id]]
    test_records = [record for ep_id in test_ids for record in episodes[ep_id]]
    
    print(f"Dataset split: {len(train_records)} train frames, {len(val_records)} val frames, {len(test_records)} test frames.")
    
    return train_records, val_records, test_records
