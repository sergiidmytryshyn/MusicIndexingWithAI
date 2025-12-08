import argparse
import os
import json
from dotenv import load_dotenv
from parser_utils import parse_info

DEFAULT_ARTIST = "Burzum"

if __name__ == "__main__":
    # Load .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Fetch artist info from Last.fm, enrich with MusicBrainz formation info (country), and save to JSON")
    parser.add_argument("--artist", default=DEFAULT_ARTIST, help="Artist name (default: Burzum)")
    parser.add_argument("--api-key", default=os.getenv("LASTFM_API_KEY"), help="Last.fm API key (or set LASTFM_API_KEY)")
    parser.add_argument("--album-limit", type=int, default=15, help="How many top albums to fetch (default: 15)")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between album calls (default: 0.5)")
    parser.add_argument("--output", default="artist_info.json", help="Output JSON file path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON in file")
    parser.add_argument(
        "--mb-user-agent",
        default=os.getenv("MB_USER_AGENT", "nlp-project/1.0 ( you@example.com )"),
        help="User-Agent for MusicBrainz (e.g., 'MyApp/1.0 ( you@example.com )' or project URL). REQUIRED by MB."
    )
    args = parser.parse_args()

    parsed_data = parse_info(args.artist, args.api_key, args.album_limit, args.sleep, args.output, args.pretty, args.mb_user_agent)

    with open(args.output, "w", encoding="utf-8") as f:
        # Serialize Pydantic model to dict for JSON (compatible with v1 and v2)
        try:
            data_dict = parsed_data.model_dump()  # Pydantic v2
        except AttributeError:
            data_dict = parsed_data.dict()  # Pydantic v1

        if args.pretty:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data_dict, f, ensure_ascii=False)
            
    print(f"✅ Data saved to {args.output}")