#!/usr/bin/env python3
"""
Convert MergedSongDataset to SongFinalDataset by chunking lyrics and descriptions,
then embedding the chunks.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.pyd_models import (
    MergedSongDataset, 
    MergedSongInfo, 
    SongFinal, 
    SongFinalDataset,
    Chunk
)

from transformers import AutoModel
from typing import List, Optional
import torch.nn as nn
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "jinaai/jina-embeddings-v3"
MODEL: nn.Module = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
MODEL.to(device)

def get_embeddings(
    texts: List[str],
    embedding_size: Optional[int] = None,
    batch_size: int = 32
) -> List[List[float]]:
    """
    Generate embeddings for a list of strings using jinaai/jina-embeddings-v3.
    
    Args:
        texts: List of strings to embed
        embedding_size: Optional embedding size. If provided, uses matryoshka encoding
                        (truncate_dim) to return embeddings of the specified size.
        batch_size: Batch size for processing texts (default: 32)
    
    Returns:
        List of embeddings, where each embedding is a list of floats
    """
        
    all_embeddings = []
    
    # Process texts in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts", total=len(texts) // batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Use model's encode method with optional truncate_dim for matryoshka encoding
        if embedding_size is not None:
            batch_embeddings = MODEL.encode(batch_texts, truncate_dim=embedding_size)
        else:
            batch_embeddings = MODEL.encode(batch_texts)
        
        # Convert numpy arrays to list of lists of floats
        all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
    
    return all_embeddings



# Configuration
INPUT_DATASET = "./data/ultra_balanced_merged_cleaned.json"
OUTPUT_DATASET = "./data/ultra_balanced_merged_chunked_embedded.json"

# Chunk sizes (in characters, approximate tokens = chars / 4)
# Based on EDA: lyrics median ~1234 chars, descriptions median ~507 chars
LYRICS_CHUNK_SIZE = 512  # ~128 tokens
DESCRIPTION_CHUNK_SIZE = 256  # ~64 tokens
CHUNK_OVERLAP = 75  # Overlap in characters (~16-17 tokens)

# Batch size for embeddings
EMBEDDING_BATCH_SIZE = 256

# Initialize LangChain text splitters for RAG-optimized chunking
# RecursiveCharacterTextSplitter respects paragraph, sentence, and word boundaries
# Note: For lyrics, we prioritize sentence/word boundaries over newlines to ensure overlap works
lyrics_splitter = RecursiveCharacterTextSplitter(
    chunk_size=LYRICS_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", ". ", "! ", "? ", "\n", " ", ""]  # Prioritize sentences over single newlines
)

description_splitter = RecursiveCharacterTextSplitter(
    chunk_size=DESCRIPTION_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
)

def chunking_text(text: str, chunk_size: int, overlap_chars: int = 0, is_lyrics: bool = True) -> List[str]:
    """
    Chunk text using LangChain's RecursiveCharacterTextSplitter for RAG.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters (used to select splitter)
        overlap_chars: Overlap in characters (used to select splitter)
        is_lyrics: Whether this is lyrics (True) or description (False)
    
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Use appropriate splitter based on content type
    splitter = lyrics_splitter if is_lyrics else description_splitter
    chunks = splitter.split_text(text)
     # Post-process to ensure overlap is present between chunks
    # LangChain sometimes doesn't add overlap when splitting at newlines
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]  # First chunk stays as is
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            # Check if overlap already exists
            prev_end = prev_chunk[-overlap_chars:].strip() if len(prev_chunk) > overlap_chars else prev_chunk.strip()
            curr_start = curr_chunk[:min(overlap_chars * 2, len(curr_chunk))].strip()
            
            # If overlap doesn't exist, add it
            if prev_end and prev_end not in curr_start:
                # Try to find a good overlap point (prefer word boundaries)
                overlap_text = prev_chunk[-overlap_chars:] if len(prev_chunk) > overlap_chars else prev_chunk
                
                # Try to start overlap at a word boundary
                words = overlap_text.split()
                if len(words) > 3:
                    # Take last few words for cleaner overlap
                    overlap_text = " ".join(words[-3:])
                
                overlapped_chunk = overlap_text + " " + curr_chunk
                # Ensure we don't exceed chunk_size too much (allow 20% tolerance)
                if len(overlapped_chunk) <= chunk_size * 1.2:
                    overlapped_chunks.append(overlapped_chunk)
                else:
                    overlapped_chunks.append(curr_chunk)
            else:
                overlapped_chunks.append(curr_chunk)
        
        return overlapped_chunks
    
    return chunks

def process_dataset():
    """Main function to process the dataset."""
    print("="*70)
    print("  CHUNK AND EMBED DATASET")
    print("="*70)
    
    # Load dataset
    input_path = Path(__file__).parent.parent / INPUT_DATASET.replace("./data/", "")
    print(f"\n1. Loading dataset from: {input_path}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset = MergedSongDataset(**data)
    print(f"✅ Loaded {len(dataset.songs):,} songs")
    
    # Step 1: Chunk all lyrics and descriptions using LangChain
    print(f"\n2. Chunking lyrics and descriptions (using LangChain RecursiveCharacterTextSplitter)...")
    print(f"   Lyrics chunk size: {LYRICS_CHUNK_SIZE} chars (~{LYRICS_CHUNK_SIZE//4} tokens)")
    print(f"   Description chunk size: {DESCRIPTION_CHUNK_SIZE} chars (~{DESCRIPTION_CHUNK_SIZE//4} tokens)")
    print(f"   Overlap: {CHUNK_OVERLAP} chars (~{CHUNK_OVERLAP//4} tokens)")
    print(f"   Strategy: Paragraphs → Sentences → Words (respects boundaries)")
    
    # Store chunks per song/artist
    lyrics_chunks_per_song: Dict[int, List[str]] = {}  # song_idx -> list of chunks
    description_chunks_cache: Dict[str, List[str]] = {}  # artist_name -> chunks
    
    for song_idx, song in enumerate(tqdm(dataset.songs, desc="Chunking")):
        # Chunk lyrics
        if song.lyrics:
            lyrics_chunks = chunking_text(song.lyrics, LYRICS_CHUNK_SIZE, CHUNK_OVERLAP, is_lyrics=True)
            lyrics_chunks_per_song[song_idx] = lyrics_chunks
        
        # Chunk description (cache by artist)
        if song.artist and song.artist.description:
            artist_name = song.artist.name
            if artist_name not in description_chunks_cache:
                desc_chunks = chunking_text(
                    song.artist.description, 
                    DESCRIPTION_CHUNK_SIZE, 
                    CHUNK_OVERLAP,
                    is_lyrics=False
                )
                description_chunks_cache[artist_name] = desc_chunks
    
    total_lyrics_chunks = sum(len(chunks) for chunks in lyrics_chunks_per_song.values())
    print(f"✅ Created {total_lyrics_chunks:,} lyrics chunks across {len(lyrics_chunks_per_song):,} songs")
    print(f"✅ Cached {len(description_chunks_cache):,} unique artist descriptions")
    
    # Step 2: Collect all unique chunks for embedding
    print(f"\n3. Preparing chunks for embedding...")
    all_chunks_to_embed: List[str] = []
    chunk_mapping: Dict[str, int] = {}  # (chunk_text) -> embedding_idx
    
    # Add lyrics chunks (deduplicate)
    for song_idx, chunks in lyrics_chunks_per_song.items():
        for chunk_text in chunks:
            if chunk_text not in chunk_mapping:
                chunk_mapping[chunk_text] = len(all_chunks_to_embed)
                all_chunks_to_embed.append(chunk_text)
    
    # Add description chunks (deduplicate)
    desc_chunk_mapping: Dict[str, int] = {}  # (chunk_text) -> embedding_idx
    for artist_name, chunks in description_chunks_cache.items():
        for chunk_text in chunks:
            if chunk_text not in desc_chunk_mapping:
                desc_chunk_mapping[chunk_text] = len(all_chunks_to_embed)
                all_chunks_to_embed.append(chunk_text)
    
    print(f"✅ Total unique chunks to embed: {len(all_chunks_to_embed):,}")
    print(f"   - Unique lyrics chunks: {len(chunk_mapping):,}")
    print(f"   - Unique description chunks: {len(desc_chunk_mapping):,}")
    
    # Step 3: Embed all chunks in batches
    print(f"\n4. Embedding chunks (batch size: {EMBEDDING_BATCH_SIZE})...")
    all_embeddings = get_embeddings(
        all_chunks_to_embed,
        batch_size=EMBEDDING_BATCH_SIZE
    )
    
    print(f"✅ Generated {len(all_embeddings):,} embeddings")
    
    # Step 4: Create SongFinal objects
    print(f"\n5. Creating SongFinal objects...")
    final_songs: List[SongFinal] = []
    
    # Create SongFinal objects
    for song_idx, song in enumerate(tqdm(dataset.songs, desc="Creating SongFinal")):
        # Get lyrics chunks
        lyrics_chunks: List[Chunk] = []
        if song_idx in lyrics_chunks_per_song:
            lyrics_chunks_list = lyrics_chunks_per_song[song_idx]
            for chunk_idx, chunk_text in enumerate(lyrics_chunks_list):
                embedding_idx = chunk_mapping[chunk_text]
                embedding = all_embeddings[embedding_idx]
                lyrics_chunks.append(Chunk(
                    idx=chunk_idx,
                    text=chunk_text,
                    embedding=embedding
                ))
        
        # Get description chunks (reuse cached chunks for artist)
        description_chunks: List[Chunk] = []
        if song.artist and song.artist.name in description_chunks_cache:
            artist_name = song.artist.name
            desc_chunks_list = description_chunks_cache[artist_name]
            for chunk_idx, chunk_text in enumerate(desc_chunks_list):
                embedding_idx = desc_chunk_mapping[chunk_text]
                embedding = all_embeddings[embedding_idx]
                description_chunks.append(Chunk(
                    idx=chunk_idx,
                    text=chunk_text,
                    embedding=embedding
                ))
        
        # Create SongFinal
        final_song = SongFinal(
            title=song.title,
            tag=song.tag,
            artist_name=song.artist_name,
            year=song.year,
            views=song.views,
            features=song.features,
            lyrics=song.lyrics,
            id=song.id,
            artist=song.artist,
            album=song.album,
            lyrics_chunks=lyrics_chunks,
            description_chunks=description_chunks
        )
        final_songs.append(final_song)
    
    print(f"✅ Created {len(final_songs):,} SongFinal objects")
    
    # Step 5: Save dataset
    print(f"\n6. Saving dataset...")
    final_dataset = SongFinalDataset(songs=final_songs)
    
    output_path = Path(__file__).parent.parent / OUTPUT_DATASET.replace("./data/", "")
    
    try:
        data_dict = final_dataset.model_dump()
    except AttributeError:
        data_dict = final_dataset.dict()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved dataset to: {output_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"Total songs processed: {len(final_songs):,}")
    total_lyrics_chunks = sum(len(song.lyrics_chunks) for song in final_songs)
    total_desc_chunks = sum(len(song.description_chunks) for song in final_songs)
    print(f"Total lyrics chunks: {total_lyrics_chunks:,}")
    print(f"Total description chunks: {total_desc_chunks:,}")
    print(f"Unique artists: {len(description_chunks_cache):,}")
    print(f"{'='*70}")

if __name__ == "__main__":
    process_dataset()

