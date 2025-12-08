#!/usr/bin/env python3
"""
Merge Last.fm/MusicBrainz artist information with Genius song datasets.
"""

import json
import time
import os
from pathlib import Path
import sys
from tqdm import tqdm
from dotenv import load_dotenv

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.pyd_models import GeniusSongDataset, GeniusSongInfo, MergedSongInfo, MergedSongDataset, ArtistInfo, AlbumInfo
from utils.parser_utils import parse_info

# Load environment variables
load_dotenv()

# File paths (only 50 songs dataset)
input_dataset = "./data/ds2_sample_10000.json"
parsed_artists_cache = "./data/parsed_artists_cache.json"
output_dataset = "./data/ds2_merged_10000.json"

# Configuration
TIMEOUT_BETWEEN_REQUESTS = 5  # seconds
ALBUM_LIMIT = 15
MB_USER_AGENT = os.getenv("MB_USER_AGENT", "nlp-project/1.0 ( you@example.com )")
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")

if not LASTFM_API_KEY:
    raise ValueError("LASTFM_API_KEY environment variable not set. Please set it in .env file.")

def load_genius_dataset():
    """Load the Genius song dataset."""
    if not Path(input_dataset).exists():
        raise FileNotFoundError(f"Dataset not found: {input_dataset}")
    
    with open(input_dataset, 'r', encoding='utf-8') as f:
        data = json.load(f)
        dataset = GeniusSongDataset(**data)
    return dataset

def get_unique_artists(dataset):
    """Extract unique artist names from dataset."""
    artists = set()
    for song in dataset.songs:
        if song.artist:
            artists.add(song.artist.strip())
    return sorted(list(artists))

def load_parsed_artists_cache():
    """Load cached parsed artist information."""
    if Path(parsed_artists_cache).exists():
        with open(parsed_artists_cache, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert dict back to ArtistInfo objects
            cache = {}
            for artist_name, artist_data in data.items():
                try:
                    cache[artist_name] = ArtistInfo(**artist_data)
                except Exception as e:
                    print(f"⚠️  Warning: Could not parse cached artist {artist_name}: {e}")
            return cache
    return {}

def save_parsed_artists_cache(cache):
    """Save parsed artist information to cache."""
    # Convert ArtistInfo objects to dicts
    cache_dict = {}
    for artist_name, artist_info in cache.items():
        try:
            cache_dict[artist_name] = artist_info.model_dump()
        except AttributeError:
            cache_dict[artist_name] = artist_info.dict()
    
    with open(parsed_artists_cache, 'w', encoding='utf-8') as f:
        json.dump(cache_dict, f, indent=2, ensure_ascii=False)

def parse_artists(artist_names, cache):
    """Parse artist information using parser_utils."""
    print(f"\nParsing {len(artist_names)} unique artists...")
    print(f"Timeout between requests: {TIMEOUT_BETWEEN_REQUESTS} seconds")
    
    parsed_count = 0
    skipped_count = 0
    
    for artist_name in tqdm(artist_names, desc="Parsing artists"):
        # Skip if already in cache
        if artist_name in cache:
            skipped_count += 1
            continue
        
        try:
            # Parse artist info (now returns ArtistInfo directly)
            artist_info = parse_info(
                artist=artist_name,
                api_key=LASTFM_API_KEY,
                album_limit=ALBUM_LIMIT,
                sleep=0.5,  # Sleep between album requests (internal)
                output="",  # Not used
                pretty=False,  # Not used
                mb_user_agent=MB_USER_AGENT
            )
            
            # Store artist info with albums
            cache[artist_name] = artist_info
            
            parsed_count += 1
            
            # Save cache after each successful parse (in case of interruption)
            save_parsed_artists_cache(cache)
            
            # Wait before next request
            if parsed_count < len(artist_names) - skipped_count:
                time.sleep(TIMEOUT_BETWEEN_REQUESTS)
                
        except Exception as e:
            print(f"\n⚠️  Error parsing {artist_name}: {e}")
            # Create minimal artist info on error
            cache[artist_name] = ArtistInfo(
                name=artist_name,
                mbid=None,
                url=None,
                description="",
                tags=[],
                founded_year=None,
                founded_place=None,
                albums=[]
            )
            save_parsed_artists_cache(cache)
            time.sleep(TIMEOUT_BETWEEN_REQUESTS)
    
    print(f"\n✅ Parsed {parsed_count} artists, {skipped_count} from cache")
    return cache

def find_album_for_song(song_title, artist_info):
    """Try to find matching album for a song in artist's albums."""
    if not artist_info or not artist_info.albums:
        return None
    
    # Simple matching: check if song title appears in any album's tracks
    song_title_lower = song_title.lower().strip()
    
    # First try exact match
    for album in artist_info.albums:
        for track in album.tracks:
            if track.name and track.name.lower().strip() == song_title_lower:
                return album
    
    # Then try partial match (song title contains track name or vice versa)
    for album in artist_info.albums:
        for track in album.tracks:
            if track.name:
                track_name_lower = track.name.lower().strip()
                if song_title_lower in track_name_lower or track_name_lower in song_title_lower:
                    return album
    
    # If no match, return None
    return None

def merge_dataset(dataset, artist_cache: dict[str, ArtistInfo]):
    """Merge Genius dataset with parsed artist information."""
    print(f"\nMerging dataset...")
    merged_songs = []
    
    for song in tqdm(dataset.songs, desc="Merging songs"):
        # Get artist info from cache
        artist_info = artist_cache.get(song.artist.strip())
        
        # Try to find matching album
        album_info = None
        if artist_info:
            album_info = find_album_for_song(song.title, artist_info)
        
        # Create merged song
        merged_song = MergedSongInfo(
            title=song.title,
            tag=song.tag,
            artist_name=song.artist,
            year=song.year,
            views=song.views,
            features=song.features,
            lyrics=song.lyrics,
            id=song.id,
            artist=artist_info,
            album=album_info
        )
        merged_songs.append(merged_song)
    
    merged_dataset = MergedSongDataset(songs=merged_songs)
    print(f"✅ Merged {len(merged_songs)} songs")
    
    return merged_dataset

def save_merged_dataset(merged_dataset):
    """Save merged dataset to JSON file."""
    print(f"\nSaving {output_dataset}...")
    
    # Convert to dict (compatible with Pydantic v1 and v2)
    try:
        data_dict = merged_dataset.model_dump()
    except AttributeError:
        data_dict = merged_dataset.dict()
    
    with open(output_dataset, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(merged_dataset.songs)} songs to {output_dataset}")

def main():
    print("="*60)
    print("MERGE ARTIST INFORMATION WITH GENIUS DATASET (50 songs)")
    print("="*60)
    
    # Load Genius dataset
    print("\n1. Loading Genius dataset...")
    dataset = load_genius_dataset()
    print(f"✅ Loaded dataset with {len(dataset.songs)} songs")
    
    # Extract unique artists
    print("\n2. Extracting unique artists...")
    unique_artists = get_unique_artists(dataset)
    print(f"✅ Found {len(unique_artists)} unique artists")
    
    # Load cache
    print("\n3. Loading parsed artists cache...")
    artist_cache = load_parsed_artists_cache()
    print(f"✅ Loaded {len(artist_cache)} cached artists")
    
    # Parse missing artists
    print("\n4. Parsing missing artists...")
    artists_to_parse = [a for a in unique_artists if a not in artist_cache]
    if artists_to_parse:
        artist_cache = parse_artists(artists_to_parse, artist_cache)
    else:
        print("✅ All artists already in cache!")
    
    # Merge dataset
    print("\n5. Merging dataset...")
    merged_dataset = merge_dataset(dataset, artist_cache)
    
    # Save merged dataset
    print("\n6. Saving merged dataset...")
    save_merged_dataset(merged_dataset)
    
    print("\n" + "="*60)
    print("MERGE COMPLETE!")
    print("="*60)
    print(f"Processed {len(unique_artists)} artists")
    print(f"Created merged dataset with {len(merged_dataset.songs)} songs")

if __name__ == "__main__":
    main()

