#!/usr/bin/env python3
"""
Sample rows from filtered dataset and save as JSON file.
"""

import pandas as pd
import json
import random
from pathlib import Path
import sys

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.pyd_models import GeniusSongInfo, GeniusSongDataset

# Configuration
SUBSET_SIZE = 10000  # Set to None to use the whole dataset, or an integer for subset size
SEED = 42  # Seed for reproducibility

# File paths
input_csv = "./data/ds2_filtered_top_artists.csv"
output_json = "./data/ds2_sample_10000.json"

# Set random seed
random.seed(SEED)

print("Loading filtered dataset...")
df = pd.read_csv(input_csv, engine='python', on_bad_lines='skip')
print(f"✅ Loaded {len(df):,} rows")

# Convert views to numeric, coercing errors to NaN
df['views'] = pd.to_numeric(df['views'], errors='coerce')

# Sample subset if specified
if SUBSET_SIZE is None:
    print("\nUsing entire dataset (SUBSET_SIZE=None)...")
    df_subset = df.copy()
else:
    if len(df) < SUBSET_SIZE:
        raise ValueError(f"Dataset has only {len(df)} rows, need at least {SUBSET_SIZE} for sampling")
    print(f"\nSampling {SUBSET_SIZE} rows (seed={SEED})...")
    df_subset = df.sample(n=SUBSET_SIZE, random_state=SEED).reset_index(drop=True)
    print(f"✅ Sampled {SUBSET_SIZE} rows")

def convert_to_pydantic(df_subset):
    """Convert DataFrame to list of GeniusSongInfo Pydantic models."""
    songs = []
    for _, row in df_subset.iterrows():
        # Handle year conversion
        year = None
        if pd.notna(row['year']):
            try:
                year = int(float(row['year']))
            except (ValueError, TypeError):
                year = None
        
        # Handle views conversion
        views = None
        if pd.notna(row['views']):
            try:
                views = int(float(row['views']))
            except (ValueError, TypeError):
                views = None
        
        # Handle id conversion
        song_id = None
        if pd.notna(row['id']):
            try:
                song_id = int(float(row['id']))
            except (ValueError, TypeError):
                song_id = None
        
        song = GeniusSongInfo(
            title=str(row['title']) if pd.notna(row['title']) else "",
            tag=str(row['tag']) if pd.notna(row['tag']) else "",
            artist=str(row['artist']) if pd.notna(row['artist']) else "",
            year=year,
            views=views,
            features=str(row['features']) if pd.notna(row['features']) else None,
            lyrics=str(row['lyrics']) if pd.notna(row['lyrics']) else None,
            id=song_id
        )
        songs.append(song)
    return songs

# Convert to Pydantic models and create dataset
print("\nConverting to Pydantic models...")
songs = convert_to_pydantic(df_subset)
dataset = GeniusSongDataset(songs=songs)

# Save as JSON
print(f"\nSaving to {output_json}...")
try:
    data_dict = dataset.model_dump()  # Pydantic v2
except AttributeError:
    data_dict = dataset.dict()  # Pydantic v1

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(songs)} songs to {output_json}")

print("\n" + "="*60)
print("SAMPLING COMPLETE")
print("="*60)
print(f"Dataset size: {len(songs)} songs")
if SUBSET_SIZE is not None:
    print(f"Subset size: {SUBSET_SIZE}")
    print(f"Seed used: {SEED}")
else:
    print("Used entire dataset (SUBSET_SIZE=None)")

