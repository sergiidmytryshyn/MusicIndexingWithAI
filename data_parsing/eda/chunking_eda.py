#!/usr/bin/env python3
"""
Exploratory Data Analysis for lyrics and artist descriptions
to understand chunking requirements.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional
import statistics
from collections import Counter

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.pyd_models import MergedSongDataset, MergedSongInfo

def count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())

def count_tokens_approx(text: str) -> int:
    """
    Approximate token count (rough estimate: 1 token ≈ 4 characters or 0.75 words).
    This is a simple heuristic - actual tokenization depends on the model.
    """
    if not text:
        return 0
    # Rough estimate: average English word is ~4.5 chars, tokens are ~4 chars
    return len(text) // 4

def analyze_text_lengths(texts: List[Optional[str]], name: str) -> dict:
    """Analyze text length statistics."""
    # Filter out None and empty strings
    valid_texts = [t for t in texts if t and t.strip()]
    empty_count = len(texts) - len(valid_texts)
    
    if not valid_texts:
        return {
            'name': name,
            'total_count': len(texts),
            'valid_count': 0,
            'empty_count': empty_count,
            'empty_percentage': 100.0,
            'error': 'No valid texts found'
        }
    
    char_lengths = [len(t) for t in valid_texts]
    word_lengths = [count_words(t) for t in valid_texts]
    token_lengths = [count_tokens_approx(t) for t in valid_texts]
    
    def get_stats(values: List[int]) -> dict:
        """Calculate statistics for a list of values."""
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'p25': statistics.quantiles(values, n=4)[0] if len(values) > 1 else values[0],
            'p75': statistics.quantiles(values, n=4)[2] if len(values) > 1 else values[0],
            'p90': statistics.quantiles(values, n=10)[8] if len(values) > 9 else max(values),
            'p95': statistics.quantiles(values, n=20)[18] if len(values) > 19 else max(values),
            'p99': statistics.quantiles(values, n=100)[98] if len(values) > 99 else max(values),
        }
    
    return {
        'name': name,
        'total_count': len(texts),
        'valid_count': len(valid_texts),
        'empty_count': empty_count,
        'empty_percentage': (empty_count / len(texts)) * 100 if texts else 0,
        'characters': get_stats(char_lengths),
        'words': get_stats(word_lengths),
        'tokens_approx': get_stats(token_lengths),
    }

def print_statistics(stats: dict):
    """Print formatted statistics."""
    print(f"\n{'='*70}")
    print(f"  {stats['name'].upper()}")
    print(f"{'='*70}")
    print(f"Total entries: {stats['total_count']:,}")
    print(f"Valid entries: {stats['valid_count']:,}")
    print(f"Empty/None entries: {stats['empty_count']:,} ({stats['empty_percentage']:.1f}%)")
    
    if 'error' in stats:
        print(f"\n⚠️  {stats['error']}")
        return
    
    print(f"\n📏 CHARACTER LENGTH STATISTICS:")
    char_stats = stats['characters']
    print(f"  Mean:     {char_stats['mean']:>10,.1f} chars")
    print(f"  Median:   {char_stats['median']:>10,.1f} chars")
    print(f"  Min:      {char_stats['min']:>10,} chars")
    print(f"  Max:      {char_stats['max']:>10,} chars")
    print(f"  Std Dev:  {char_stats['std']:>10,.1f} chars")
    print(f"  P25:      {char_stats['p25']:>10,.1f} chars")
    print(f"  P75:      {char_stats['p75']:>10,.1f} chars")
    print(f"  P90:      {char_stats['p90']:>10,.1f} chars")
    print(f"  P95:      {char_stats['p95']:>10,.1f} chars")
    print(f"  P99:      {char_stats['p99']:>10,.1f} chars")
    
    print(f"\n📝 WORD COUNT STATISTICS:")
    word_stats = stats['words']
    print(f"  Mean:     {word_stats['mean']:>10,.1f} words")
    print(f"  Median:   {word_stats['median']:>10,.1f} words")
    print(f"  Min:      {word_stats['min']:>10,} words")
    print(f"  Max:      {word_stats['max']:>10,} words")
    print(f"  Std Dev:  {word_stats['std']:>10,.1f} words")
    print(f"  P25:      {word_stats['p25']:>10,.1f} words")
    print(f"  P75:      {word_stats['p75']:>10,.1f} words")
    print(f"  P90:      {word_stats['p90']:>10,.1f} words")
    print(f"  P95:      {word_stats['p95']:>10,.1f} words")
    print(f"  P99:      {word_stats['p99']:>10,.1f} words")
    
    print(f"\n🎯 APPROXIMATE TOKEN COUNT STATISTICS:")
    token_stats = stats['tokens_approx']
    print(f"  Mean:     {token_stats['mean']:>10,.1f} tokens")
    print(f"  Median:   {token_stats['median']:>10,.1f} tokens")
    print(f"  Min:      {token_stats['min']:>10,} tokens")
    print(f"  Max:      {token_stats['max']:>10,} tokens")
    print(f"  Std Dev:  {token_stats['std']:>10,.1f} tokens")
    print(f"  P25:      {token_stats['p25']:>10,.1f} tokens")
    print(f"  P75:      {token_stats['p75']:>10,.1f} tokens")
    print(f"  P90:      {token_stats['p90']:>10,.1f} tokens")
    print(f"  P95:      {token_stats['p95']:>10,.1f} tokens")
    print(f"  P99:      {token_stats['p99']:>10,.1f} tokens")

def analyze_chunking_recommendations(lyrics_stats: dict, desc_stats: dict):
    """Provide recommendations for chunking based on statistics."""
    print(f"\n{'='*70}")
    print(f"  CHUNKING RECOMMENDATIONS")
    print(f"{'='*70}")
    
    # Common chunk sizes (in tokens)
    # Most embedding models have limits: OpenAI ada-002: 8191, text-embedding-3: 8191
    # Sentence transformers typically: 512-1024 tokens
    chunk_sizes = {
        'small': 128,   # ~512 chars, ~100 words
        'medium': 256,  # ~1024 chars, ~200 words
        'large': 512,   # ~2048 chars, ~400 words
        'xlarge': 1024, # ~4096 chars, ~800 words
    }
    
    print("\n💡 RECOMMENDED CHUNK SIZES (in approximate tokens):")
    print("   - Small (128 tokens):  Good for short descriptions, preserves granularity")
    print("   - Medium (256 tokens): Balanced for most use cases")
    print("   - Large (512 tokens):  Good for longer texts, reduces chunk count")
    print("   - XLarge (1024 tokens): Maximum for most embedding models")
    
    if 'characters' in lyrics_stats:
        print(f"\n📊 LYRICS CHUNKING ANALYSIS:")
        lyrics_median = lyrics_stats['characters']['median']
        lyrics_p75 = lyrics_stats['characters']['p75']
        lyrics_p90 = lyrics_stats['characters']['p90']
        
        print(f"   Median lyrics length: {lyrics_median:,.0f} chars")
        print(f"   75th percentile: {lyrics_p75:,.0f} chars")
        print(f"   90th percentile: {lyrics_p90:,.0f} chars")
        
        # Estimate chunks needed
        for size_name, size_tokens in chunk_sizes.items():
            size_chars = size_tokens * 4  # Rough conversion
            chunks_median = max(1, int(lyrics_median / size_chars))
            chunks_p90 = max(1, int(lyrics_p90 / size_chars))
            print(f"   - {size_name.capitalize()} chunks ({size_chars} chars): "
                  f"~{chunks_median} chunks (median), ~{chunks_p90} chunks (p90)")
    
    if 'characters' in desc_stats:
        print(f"\n📊 ARTIST DESCRIPTION CHUNKING ANALYSIS:")
        desc_median = desc_stats['characters']['median']
        desc_p75 = desc_stats['characters']['p75']
        desc_p90 = desc_stats['characters']['p90']
        
        print(f"   Median description length: {desc_median:,.0f} chars")
        print(f"   75th percentile: {desc_p75:,.0f} chars")
        print(f"   90th percentile: {desc_p90:,.0f} chars")
        
        # Estimate chunks needed
        for size_name, size_tokens in chunk_sizes.items():
            size_chars = size_tokens * 4  # Rough conversion
            chunks_median = max(1, int(desc_median / size_chars))
            chunks_p90 = max(1, int(desc_p90 / size_chars))
            print(f"   - {size_name.capitalize()} chunks ({size_chars} chars): "
                  f"~{chunks_median} chunks (median), ~{chunks_p90} chunks (p90)")

def main():
    """Main EDA function."""
    # File path
    dataset_path = Path(__file__).parent.parent / "data" / "ultra_balanced_merged_cleaned.json"
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Available files:")
        data_dir = Path(__file__).parent.parent / "data"
        for f in data_dir.glob("*.json"):
            print(f"  - {f.name}")
        return
    
    print("="*70)
    print("  EXPLORATORY DATA ANALYSIS: CHUNKING REQUIREMENTS")
    print("="*70)
    print(f"\nLoading dataset from: {dataset_path}")
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset = MergedSongDataset(**data)
    print(f"✅ Loaded {len(dataset.songs):,} songs\n")
    
    # Extract lyrics and descriptions
    lyrics = [song.lyrics for song in dataset.songs]
    descriptions = []
    
    for song in dataset.songs:
        if song.artist and song.artist.description:
            descriptions.append(song.artist.description)
        else:
            descriptions.append(None)
    
    # Analyze lyrics
    lyrics_stats = analyze_text_lengths(lyrics, "Lyrics")
    print_statistics(lyrics_stats)
    
    # Analyze artist descriptions
    desc_stats = analyze_text_lengths(descriptions, "Artist Descriptions")
    print_statistics(desc_stats)
    
    # Chunking recommendations
    analyze_chunking_recommendations(lyrics_stats, desc_stats)
    
    # Additional insights
    print(f"\n{'='*70}")
    print(f"  ADDITIONAL INSIGHTS")
    print(f"{'='*70}")
    
    # Songs with both lyrics and descriptions
    songs_with_both = sum(1 for song in dataset.songs 
                          if song.lyrics and song.artist and song.artist.description)
    print(f"\nSongs with both lyrics AND artist description: {songs_with_both:,} "
          f"({songs_with_both/len(dataset.songs)*100:.1f}%)")
    
    # Songs with lyrics but no description
    songs_lyrics_only = sum(1 for song in dataset.songs 
                            if song.lyrics and (not song.artist or not song.artist.description))
    print(f"Songs with lyrics but NO description: {songs_lyrics_only:,} "
          f"({songs_lyrics_only/len(dataset.songs)*100:.1f}%)")
    
    # Songs with description but no lyrics
    songs_desc_only = sum(1 for song in dataset.songs 
                          if (not song.lyrics) and song.artist and song.artist.description)
    print(f"Songs with description but NO lyrics: {songs_desc_only:,} "
          f"({songs_desc_only/len(dataset.songs)*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

