"""
Script to generate diverse prompts for finding tracks using OLLAMA.
"""

import json
import random
from typing import Dict, List

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    exit(1)

# Configuration
SAMPLES_FILE = "data_parsing/data/sample_100.json"
OUTPUT_FILE = "data_parsing/data/generated_prompts.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:14b"


def call_ollama(prompt: str) -> str:
    """Call OLLAMA API to generate text."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Error calling OLLAMA: {e}")
        return ""


def extract_song_info(song: Dict) -> Dict:
    """Extract relevant information from a song for prompt generation."""
    info = {
        "title": song.get("title", ""),
        "artist_name": song.get("artist_name", ""),
        "genre": song.get("tag", ""),
        "year": song.get("year"),
        "lyrics": song.get("lyrics", ""),
        "artist_country": None,
        "artist_description": None,
        "lyrics_snippet": None
    }
    
    # Extract artist info
    artist = song.get("artist")
    if artist:
        info["artist_country"] = artist.get("founded_place")
        info["artist_description"] = artist.get("description", "")
    
    # Extract a random snippet from lyrics (if available)
    lyrics = info["lyrics"]
    if lyrics and len(lyrics) > 100:
        # Get a random snippet of 50-150 characters
        start = random.randint(0, max(0, len(lyrics) - 150))
        end = start + random.randint(50, 150)
        info["lyrics_snippet"] = lyrics[start:end].strip()
    
    return info


def generate_prompt_for_track(song_info: Dict, previous_prompts: List[str] = None) -> str:
    """Generate a prompt using OLLAMA for finding this track."""
    
    # Select only a subset of available information (randomly)
    available_info = []
    
    if song_info["title"]:
        available_info.append(("title", song_info["title"]))
    if song_info["artist_name"]:
        available_info.append(("artist", song_info["artist_name"]))
    if song_info["genre"]:
        available_info.append(("genre", song_info["genre"]))
    if song_info["year"]:
        available_info.append(("year", song_info["year"]))
    if song_info["artist_country"]:
        available_info.append(("country", song_info["artist_country"]))
    
    # Lyrics with 75% chance
    selected_info = []
    if song_info["lyrics_snippet"] and random.random() < 0.75:
        selected_info.append(("lyrics", song_info["lyrics_snippet"]))
    
    # Then add 1-2 more pieces randomly from remaining info
    remaining_info = [info for info in available_info]
    if remaining_info:
        num_additional = random.randint(2, min(3, len(remaining_info))) if selected_info else random.randint(3, min(4, len(remaining_info)))
        additional = random.sample(remaining_info, num_additional)
        selected_info.extend(additional)
    
    # Build context with selected info only
    context_parts = []
    lyrics_instruction = ""
    
    for key, value in selected_info:
        if key == "lyrics":
            # Flip a coin: exact or approximate
            use_exact = random.random() < 0.75
            if use_exact:
                context_parts.append(f"Lyrics snippet (use exact phrasing): \"{value}\"")
                lyrics_instruction = "\n- For lyrics: Use the EXACT phrasing from the lyrics snippet provided"
            else:
                context_parts.append(f"Lyrics snippet (use approximate description): \"{value}\"")
                lyrics_instruction = "\n- For lyrics: Use APPROXIMATE description of what the lyrics are about, NOT the exact words"
        elif key == "country":
            context_parts.append(f"Location (you can use either the exact country '{value}' OR its region/part of world like 'Europe', 'North America', 'Oceania', etc.): {value}")
        else:
            context_parts.append(f"{key.capitalize()}: {value}")
    
    context = "\n".join(context_parts)
    
    # Build previous prompts section
    previous_section = ""
    if previous_prompts and len(previous_prompts) > 0:
        previous_examples = "\n".join([f"{i+1}. {p}" for i, p in enumerate(previous_prompts[-3:])])
        previous_section = f"""

IMPORTANT - Avoid similar phrasings! Here are 3 previous prompts (DO NOT use similar structure/phrasing):
{previous_examples}

Your prompt must be DIFFERENT in structure, phrasing, and style from these examples. Be creative and varied."""
    
    # Create prompt for OLLAMA
    ollama_prompt = f"""You are helping create a natural language query for finding a specific song in a music search system.

Given the following information about a song, create a concise, natural query (1-2 sentences max) that a user might use to find this song. The query should:
- Be conversational and natural (like a real user searching)
- Be concise (2-3 sentences at most)
- Include only SOME of the information provided - be selective
- Mix different types of information naturally
- Sound like someone trying to remember/find a specific song
- Use varied phrasing and structure{lyrics_instruction}
- For location: You can use either the exact country OR its region/part of world (e.g., 'Europe', 'North America', 'Oceania')

Song information (use only some of this):
{context}{previous_section}

Generate a concise, natural, diverse query for finding this song:"""

    return call_ollama(ollama_prompt)


def main():
    """Main function to generate prompts for all samples."""
    print(f"Loading samples from: {SAMPLES_FILE}")
    
    with open(SAMPLES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    songs = data.get('songs', [])
    print(f"Found {len(songs)} songs\n")
    
    results = []
    previous_prompts = []
    
    for i, song in enumerate(songs, 1):
        print(f"[{i}/{len(songs)}] Processing: {song.get('title', 'Unknown')} - {song.get('artist_name', 'Unknown')}")
        
        # Extract relevant info
        song_info = extract_song_info(song)
        
        # Generate prompt with previous prompts for diversity
        generated_prompt = generate_prompt_for_track(song_info, previous_prompts)
        
        if not generated_prompt:
            print(f"  ⚠️  Failed to generate prompt, skipping...")
            continue
        
        # Store result
        result = {
            "track_id": song.get("id"),
            "title": song_info["title"],
            "artist_name": song_info["artist_name"],
            "genre": song_info["genre"],
            "year": song_info["year"],
            "generated_prompt": generated_prompt,
            "metadata": {
                "has_lyrics": bool(song_info["lyrics"]),
                "has_artist_info": bool(song_info["artist_description"]),
                "artist_country": song_info["artist_country"]
            }
        }
        
        results.append(result)
        previous_prompts.append(generated_prompt)  # Track for next iterations
        print(f"  ✅ Generated: {generated_prompt[:80]}...")
    
    # Save results
    output_data = {
        "total_tracks": len(results),
        "prompts": results
    }
    
    print(f"\nSaving {len(results)} prompts to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Done! Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

