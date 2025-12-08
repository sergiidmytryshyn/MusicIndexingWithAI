#!/usr/bin/env python3
"""
Filter ds2_full.csv to only include rows for artists in top_10_artists_per_genre.csv
"""

import pandas as pd

# File paths (relative to script location)
input_csv = "./data/ds2_full.csv"
top_artists_csv = "./eda/top_200_artists.csv"
output_csv = "./data/ds2_filtered_top_artists.csv"

print("Loading top artists list...")
top_artists_df = pd.read_csv(top_artists_csv)

# Get unique set of artist names
top_artists = set(top_artists_df['artist'].str.strip().unique())
print(f"✅ Found {len(top_artists)} unique top artists")

print(f"\nLoading {input_csv}...")
df = pd.read_csv(input_csv, engine='python', on_bad_lines='skip')
print(f"✅ Loaded {len(df):,} rows")

# Filter: keep rows where artist is in top_artists set
print("\nFiltering rows...")
df['artist_clean'] = df['artist'].astype(str).str.strip()
filtered_df = df[df['artist_clean'].isin(top_artists)].copy()
filtered_df = filtered_df.drop(columns=['artist_clean'])

print(f"✅ Filtered to {len(filtered_df):,} rows ({len(filtered_df)/len(df)*100:.2f}% of original)")

# Convert views to numeric for sorting
print("\nLimiting to top 100 tracks per artist (by views)...")
filtered_df['views'] = pd.to_numeric(filtered_df['views'], errors='coerce')

# Group by artist and limit to top 100 by views
def limit_artist_tracks(group):
    """Limit each artist to top 100 tracks by views."""
    if len(group) > 100:
        # Sort by views descending and take top 100
        return group.nlargest(100, 'views')
    return group

filtered_df = filtered_df.groupby('artist', group_keys=False).apply(limit_artist_tracks).reset_index(drop=True)

artists_limited = filtered_df.groupby('artist').size()
limited_count = (artists_limited > 100).sum()
print(f"✅ Limited {limited_count} artists to top 100 tracks each")
print(f"Final dataset: {len(filtered_df):,} rows")

# Save to output file
print(f"\nSaving to {output_csv}...")
filtered_df.to_csv(output_csv, index=False)
print(f"✅ Filtered data saved to {output_csv}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total rows in filtered dataset: {len(filtered_df):,}")
print(f"\nTop artists in filtered data:")
artists_in_data = filtered_df['artist'].value_counts()
print(artists_in_data.head(10).to_string())
print(f"\nGenres in filtered data:")
genres_in_data = filtered_df['tag'].value_counts()
print(genres_in_data.to_string())

