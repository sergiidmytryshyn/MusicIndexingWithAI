#!/usr/bin/env python3
"""
Clean up merged dataset by removing tracks without proper artist information.
"""

import json
from pathlib import Path
import sys

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.pyd_models import MergedSongDataset, MergedSongInfo

# File paths
input_json = "./data/ds2_merged_10000.json"
output_json = "./data/ds2_merged_10000_cleaned.json"

def has_proper_artist_info(song: MergedSongInfo) -> bool:
    """
    Check if song has proper artist information.
    Returns False if artist is None or has minimal/default info.
    """
    if song.artist is None:
        return False
    
    # Check if artist has meaningful information (not just default/empty values)
    artist = song.artist
    
    # If description is empty and no albums, likely a failed parse
    if not artist.description and len(artist.albums) == 0:
        return False
    
    # If no description, no tags, no albums, and no URL - likely failed parse
    if (not artist.description and 
        len(artist.tags) == 0 and 
        len(artist.albums) == 0 and 
        not artist.url):
        return False
    
    return True

def cleanup_dataset(input_path: str, output_path: str):
    """Load merged dataset, remove tracks without proper artist info, and save cleaned version."""
    print(f"Loading merged dataset from {input_path}...")
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset = MergedSongDataset(**data)
    print(f"✅ Loaded {len(dataset.songs)} songs")
    
    # Filter out songs without proper artist info
    print("\nFiltering songs without proper artist information...")
    original_count = len(dataset.songs)
    
    cleaned_songs = [song for song in dataset.songs if has_proper_artist_info(song)]
    removed_count = original_count - len(cleaned_songs)
    
    print(f"✅ Removed {removed_count} songs without proper artist info")
    print(f"✅ Kept {len(cleaned_songs)} songs ({len(cleaned_songs)/original_count*100:.1f}% of original)")
    
    # Create cleaned dataset
    cleaned_dataset = MergedSongDataset(songs=cleaned_songs)
    
    # Save cleaned dataset
    print(f"\nSaving cleaned dataset to {output_path}...")
    try:
        data_dict = cleaned_dataset.model_dump()  # Pydantic v2
    except AttributeError:
        data_dict = cleaned_dataset.dict()  # Pydantic v1
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Cleaned dataset saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CLEANUP SUMMARY")
    print("="*60)
    print(f"Original songs: {original_count}")
    print(f"Removed songs: {removed_count}")
    print(f"Final songs: {len(cleaned_songs)}")
    print(f"Removal rate: {removed_count/original_count*100:.1f}%")

if __name__ == "__main__":
    cleanup_dataset(input_json, output_json)

