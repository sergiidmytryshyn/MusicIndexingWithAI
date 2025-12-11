"""
Benchmark script for testing field extraction from various prompts.
Generates a report file with prompts, XML responses, JSON results, and extraction issues.
"""

import json
from typing import Dict
from datetime import datetime
import sys
import os

# Add parent directory to path to import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import extraction function from app.py
# We need to mock streamlit since it's not available in CLI context
import streamlit as st
st.write = lambda *args, **kwargs: None  # Mock st.write to do nothing
st.error = lambda *args, **kwargs: None
st.warning = lambda *args, **kwargs: None
st.code = lambda *args, **kwargs: None
st.stop = lambda: None

from app import extract_fields_with_costar

# Test prompts - diverse, challenging prompts for finding specific tracks
TEST_PROMPTS = [
    # Complex multi-criteria searches
    "I'm trying to find this rock song from the 70s, I think it's British, and it has this epic guitar solo. The lyrics mention something about a stairway or heaven, and it's really long",
    "There's this French song from maybe the 40s or 50s, a woman singing, very romantic, I think the title has 'rose' in it and it's about seeing life through rose-colored glasses",
    "Looking for a Norwegian metal band, they started in the 90s, very dark sound, I think they sing about forests or nature, something atmospheric and heavy",
    
    # Lyrics-based searches (should extract lyrics_text)
    "I remember a song where the chorus goes something like 'I remember when, I remember when I lost my mind' - it's a pop or electronic song, maybe from the 2000s",
    "There's this song that talks about a man who had a farm, and it mentions animals like a cow and a pig, I think it's a children's song or maybe country",
    "I'm looking for a song that says 'we are the champions of the world' - it's a rock anthem, very powerful, probably from the 70s or 80s",
    "There's this hip hop track where the rapper talks about growing up in the streets, mentions his neighborhood, something about making it out, probably from the US",
    "I need to find a song where the lyrics mention 'tears falling down' or something about crying, it's a sad ballad, maybe pop or R&B",
    "Looking for a song that has 'highway to hell' in the lyrics, it's a classic rock song, very energetic",
    
    # Topic-based searches (should extract lyrics_text)
    "I'm trying to find songs about heartbreak, specifically about a relationship ending, maybe something about being left behind or moving on",
    "There's this song about love and relationships, talks about being together forever, I think it's a pop ballad from the 80s or 90s",
    "Looking for a song about memories, something nostalgic, talks about the good old days or remembering better times",
    
    # Characteristics only (should NOT extract lyrics_text)
    "I want something really danceable and viral, maybe K-pop from South Korea, released in the last few years, something that went viral on TikTok",
    "Looking for upbeat and catchy pop songs from Oceania, maybe Australia or New Zealand, something with tropical vibes, released in the 2010s",
    "I need motivational hip hop tracks, something powerful and inspiring, featuring artists like Eminem or Rihanna, from the United States",
    "Find me epic and powerful rock songs from British bands in the 1970s, something with big guitar riffs and dramatic vocals",
    
    # Mixed cases - lyrics + characteristics (should separate correctly)
    "There's this sad song with 'rain' in the lyrics, talks about standing in the rain, it's slow and emotional, probably a ballad",
    "I'm looking for upbeat pop songs about dancing, the lyrics mention dancing or partying, something catchy and fun, maybe from the 2010s",
    "Find me a song that says something about 'losing yourself' or 'finding yourself', it's motivational hip hop, probably from the 2000s",
    
    # Complex ambiguous cases
    "I remember a song by The Beatles, from their early period around 1960-1965, I think it's from an album with 'love' in the title, and the song itself might have 'love' in the lyrics too",
    "Looking for hip hop tracks from North America, featuring Drake, released after 2010, and the lyrics might mention something about success or money",
    "There's this song with 'hotel' in the title, it's rock, from North America, released in the 1970s, and I think the lyrics tell some kind of story about a hotel"
]


def run_benchmark():
    """Run benchmark tests on all prompts and generate report."""
    results = []
    
    print(f"Running benchmark on {len(TEST_PROMPTS)} prompts...")
    print("=" * 80)
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Processing: {prompt[:60]}...")
        
        try:
            result = extract_fields_with_costar(prompt)
            
            # Extract debug info
            debug_info = result.pop("_debug", {})
            
            # Create result entry
            entry = {
                "prompt": prompt,
                "raw_xml": debug_info.get("raw_xml", "N/A"),
                "extracted_json": result,
                "parsed_fields": debug_info.get("parsed_fields", []),
                "missing_fields": debug_info.get("missing_fields", []),
                "parsing_errors": debug_info.get("parsing_errors", []),
                "has_errors": len(debug_info.get("parsing_errors", [])) > 0 or len(debug_info.get("missing_fields", [])) > 0
            }
            
            results.append(entry)
            
            # Print summary
            if entry["has_errors"]:
                print(f"  ⚠️  Issues found: {len(entry['parsing_errors'])} errors, {len(entry['missing_fields'])} missing fields")
            else:
                print(f"  ✅ Successfully extracted")
                
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            results.append({
                "prompt": prompt,
                "raw_xml": "N/A",
                "extracted_json": {},
                "parsed_fields": [],
                "missing_fields": ["Benchmark execution failed"],
                "parsing_errors": [f"Exception: {str(e)}"],
                "has_errors": True
            })
    
    # Generate report file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"benchmark_report_{timestamp}.json"
    
    report = {
        "timestamp": timestamp,
        "total_prompts": len(TEST_PROMPTS),
        "results": results,
        "summary": {
            "total_tests": len(results),
            "successful": len([r for r in results if not r["has_errors"]]),
            "with_errors": len([r for r in results if r["has_errors"]]),
            "total_parsing_errors": sum(len(r["parsing_errors"]) for r in results),
            "total_missing_fields": sum(len(r["missing_fields"]) for r in results)
        }
    }
    
    # Save report
    with open(report_filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print(f"\n✅ Benchmark complete!")
    print(f"📊 Summary:")
    print(f"   Total tests: {report['summary']['total_tests']}")
    print(f"   Successful: {report['summary']['successful']}")
    print(f"   With errors: {report['summary']['with_errors']}")
    print(f"   Total parsing errors: {report['summary']['total_parsing_errors']}")
    print(f"   Total missing fields: {report['summary']['total_missing_fields']}")
    print(f"\n📄 Report saved to: {report_filename}")
    
    return report_filename


if __name__ == "__main__":
    run_benchmark()

