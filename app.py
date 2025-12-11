import streamlit as st
import os
from anthropic import Anthropic
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
import json
from typing import Optional, Dict, List

load_dotenv()

# Initialize Anthropic client
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("Please set ANTHROPIC_API_KEY in your environment variables or .env file")
    st.stop()

client = Anthropic(api_key=api_key)

# Page configuration
st.set_page_config(
    page_title="MIWA - Music Indexing with AI",
    page_icon="🎵",
    layout="wide"
)

# Title and description
st.title("🎵 MIWA - Music Indexing with AI")
st.markdown("**Find songs by mood, lyrics, artist, location (country/region), or genre!**")
st.markdown("---")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def print_debug_info(debug_info: Dict):
    """
    Print debug information to console (not UI).
    Shows parsed fields, missing fields, parsing errors, and raw XML.
    """
    st.write("--- DEBUG INFO ---")
    
    if debug_info.get("parsed_fields"):
        st.write("✅ Successfully Parsed Fields:")
        for field in debug_info["parsed_fields"]:
            st.write(f"  • {field}")
    else:
        st.write("✅ No fields were successfully parsed")
    
    if debug_info.get("missing_fields"):
        st.write("\n❌ Missing/Not Found Fields:")
        for field in debug_info["missing_fields"]:
            st.write(f"  • {field}")
    else:
        st.write("\n❌ All expected fields were found")
    
    if debug_info.get("parsing_errors"):
        st.write("\n⚠️ Parsing Errors:")
        for error in debug_info["parsing_errors"]:
            st.write(f"  • {error}")
    
    if debug_info.get("raw_xml"):
        st.write("\n📄 Raw XML Response:")
        st.write(debug_info["raw_xml"])
    
    st.write("--- END DEBUG INFO ---\n")


def extract_fields_with_costar(user_query: str) -> Dict:
    """
    Extract detailed structured information from user queries about songs using COSTAR format.
    Returns a dictionary with track, artist, features, and album information.
    """
    
    costar_prompt = f"""<CONTEXT>
You are a music information extraction system for MIWA (Music Indexing with AI). Your task is to extract structured information from user queries about songs they want to find.

Users may mention various aspects of songs:
- Track details: title keywords, release year (ranges), genres, lyrics content
- Artist details: name keywords, founding year, genres, location (country/region), description
- Featured artists: artists who appear on tracks
- Album details: name keywords

Extract ONLY information that is explicitly mentioned or can be clearly inferred from the query. If a field is not mentioned, use null or empty values. Do not make assumptions beyond what is stated.
</CONTEXT>

<OBJECTIVE>
Extract structured information from the user query into the following fields:

**TRACK INFORMATION:**
- title_keywords: List of keywords from the song title mentioned (e.g., ["love", "heart"] if user says "songs with 'love' in the title")
- year_from: Starting year if a range is mentioned (e.g., "songs from 2000-2010" → year_from=2000)
- year_to: Ending year if a range is mentioned (e.g., "songs from 2000-2010" → year_to=2010)
- genres: List of genres mentioned (e.g., ["rock", "pop"]). Use lowercase.
- lyrics_keywords: List of keywords from lyrics mentioned (e.g., ["rain", "tears"] if user mentions these words)
- lyrics_text: A descriptive query for when the user mentions actual lyrics content, parts of lyrics, approximate lyrics or song topic (e.g., "songs with 'tears' in the lyrics" → "tears and crying", "I remember when..." → "songs about memories", "song about a man who had a farm  " → "a man who had a farm"). This is NOT exact lyrics, but a description of the lyrical content referenced (or could be an exact lyric if explicitly mentioned)

**ARTIST INFORMATION:**
- name_keywords: List of keywords from artist name mentioned (e.g., ["beatles"] if user says "songs by The Beatles")
- founded_year_from: Starting year if artist founding year range is mentioned
- founded_year_to: Ending year if artist founding year range is mentioned
- genres: List of genres the artist performs (if mentioned separately from track genres)
- country: Country name if mentioned (e.g., "United States", "France", "United Kingdom", "Norway"). Use standard country names.
- region: Region name if mentioned instead of country (e.g., "Europe", "North America", "Oceania", "Asia", "South America"). If country is mentioned, set region to null.
- description_keywords: List of keywords about the artist's description mentioned
- description_text: A descriptive query summarizing what the artist is known for (e.g., "rock band from UK" → "rock band from United Kingdom"). This is NOT exact description text, but a description of the artist's characteristics.

**FEATURES:**
- features: List of artist names who are featured on tracks (e.g., ["Drake", "Rihanna"] if user mentions "songs featuring Drake and Rihanna")

**ALBUM INFORMATION:**
- name_keywords: List of keywords from album name mentioned

Output ONLY valid XML with no additional text, comments, or explanations.
</OBJECTIVE>

<STYLE>
- Be precise: only extract what is clearly mentioned or strongly implied
- Use lowercase for genres (e.g., "rock", "hip hop", "r&b")
- Use standard country names: "United States", "United Kingdom", "France", "Canada", "Australia", "Norway", etc.
- Use standard region names: "Europe", "North America", "South America", "Asia", "Oceania"
- For year ranges: if user says "2000s" or "early 2000s", infer reasonable ranges (e.g., 2000-2009)
- For description_text: create concise descriptive queries about artist characteristics (what they're known for, their style, origin)
- For lyrics_text: when user mentions actual lyrics, parts of lyrics, approximate lyrics or song topic. Create a descriptive query capturing the lyrical content referenced.
- Empty lists should be represented as empty lists, not null
- Numbers should be integers (no decimals for years)
</STYLE>

<AUDIENCE>
Music search system (GraphRAG) that will use these fields to query a Neo4j database containing tracks, artists, albums, genres, and locations.
</AUDIENCE>

<RESPONSE_FORMAT>
Output ONLY XML in the following format, with no other text:

<extraction>
    <track>
        <title_keywords>
            <keyword>keyword1</keyword>
            <keyword>keyword2</keyword>
        </title_keywords>
        <year_from>2000</year_from>
        <year_to>2010</year_to>
        <genres>.
            <genre>rock</genre>
            <genre>pop</genre>
        </genres>
        <lyrics_keywords>
            <keyword>love</keyword>
            <keyword>heart</keyword>
        </lyrics_keywords>
        <lyrics_text>songs about love and relationships</lyrics_text>
    </track>
    <artist>
        <name_keywords>
            <keyword>beatles</keyword>
        </name_keywords>
        <founded_year_from>1960</founded_year_from>
        <founded_year_to>1962</founded_year_to>
        <genres>
            <genre>rock</genre>
        </genres>
        <country>United Kingdom</country>
        <region>null</region>
        <description_keywords>
            <keyword>legendary</keyword>
        </description_keywords>
        <description_text>legendary rock band from United Kingdom</description_text>
    </artist>
    <features>
        <feature>Drake</feature>
        <feature>Rihanna</feature>
    </features>
    <album>
        <name_keywords>
            <keyword>thriller</keyword>
        </name_keywords>
    </album>
</extraction>

IMPORTANT: 
- If a field is not mentioned, use null (for single values) or empty tags (for lists)
- For empty lists, use empty tags like <title_keywords></title_keywords>
- Numbers should be integers only
- Use null (as text) for missing single values
</RESPONSE_FORMAT>

<REASONING>
Think step by step:
1. Analyze the query for track-related information (title, year, genres, lyrics)
2. Analyze the query for artist-related information (name, founding year, genres, location, description)
3. Identify any featured artists mentioned
4. Analyze the query for album-related information (name)
5. For lyrics_text: only extract if user mentions actual lyrics, parts of lyrics, or references to what the song says. If user only mentions musical characteristics without lyrics content, leave null/empty
6. For description_text: create descriptive queries about artist characteristics
7. Only include fields that are explicitly mentioned or clearly implied
</REASONING>

User Query: {user_query}

Now output ONLY the XML response with no additional text:"""

    def parse_list_element(parent, tag_name, item_tag="keyword"):
        """Helper to parse list elements from XML"""
        elem = parent.find(tag_name)
        if elem is None:
            return []
        items = []
        for item in elem.findall(item_tag):
            if item.text:
                items.append(item.text.strip())
        return items
    
    def parse_int_or_none(elem):
        """Helper to parse integer or None"""
        if elem is None or elem.text is None:
            return None
        text = elem.text.strip().lower()
        if text == "null" or text == "":
            return None
        try:
            return int(text)
        except ValueError:
            return None
    
    def parse_text_or_none(elem):
        """Helper to parse text or None"""
        if elem is None or elem.text is None:
            return None
        text = elem.text.strip()
        if text.lower() == "null" or text == "":
            return None
        return text
    
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": costar_prompt}
            ]
        )
        
        response_text = message.content[0].text.strip()
        
        # Parse XML response
        try:
            # Try to extract XML from the response (in case there's extra text)
            if "<extraction>" in response_text:
                start = response_text.find("<extraction>")
                end = response_text.find("</extraction>") + len("</extraction>")
                xml_text = response_text[start:end]
            else:
                xml_text = response_text
            
            # Fix common XML issues: escape unescaped & symbols
            import re
            # Replace & that is not part of an entity (&amp;, &lt;, &gt;, &quot;, &apos;)
            xml_text = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)', '&amp;', xml_text)
            
            root = ET.fromstring(xml_text)
            
            # Initialize debug tracking
            debug_info = {
                "parsed_fields": [],
                "missing_fields": [],
                "parsing_errors": [],
                "raw_xml": xml_text
            }
            
            def track_field(field_path: str, xml_elem, value, is_list: bool = False):
                """Track field parsing status for debug output."""
                if xml_elem is None:
                    debug_info["missing_fields"].append(f"{field_path}: element not found in XML")
                elif is_list:
                    if value and len(value) > 0:
                        debug_info["parsed_fields"].append(f"{field_path}: {len(value)} items")
                    else:
                        debug_info["parsed_fields"].append(f"{field_path}: empty list (present in XML)")
                else:
                    if value is not None:
                        debug_info["parsed_fields"].append(f"{field_path}: {value}")
                    else:
                        debug_info["parsed_fields"].append(f"{field_path}: null/empty (present in XML)")
            
            # Parse track information
            track_elem = root.find("track")
            track = {}
            if track_elem is not None:
                try:
                    title_keywords_elem = track_elem.find("title_keywords")
                    title_keywords = parse_list_element(track_elem, "title_keywords", "keyword")
                    track_field("track.title_keywords", title_keywords_elem, title_keywords, is_list=True)
                except Exception as e:
                    title_keywords = []
                    debug_info["parsing_errors"].append(f"track.title_keywords: {str(e)}")
                
                try:
                    year_from_elem = track_elem.find("year_from")
                    year_from = parse_int_or_none(year_from_elem)
                    track_field("track.year_from", year_from_elem, year_from)
                except Exception as e:
                    year_from = None
                    debug_info["parsing_errors"].append(f"track.year_from: {str(e)}")
                
                try:
                    year_to_elem = track_elem.find("year_to")
                    year_to = parse_int_or_none(year_to_elem)
                    track_field("track.year_to", year_to_elem, year_to)
                except Exception as e:
                    year_to = None
                    debug_info["parsing_errors"].append(f"track.year_to: {str(e)}")
                
                try:
                    genres_elem = track_elem.find("genres")
                    genres = parse_list_element(track_elem, "genres", "genre")
                    track_field("track.genres", genres_elem, genres, is_list=True)
                except Exception as e:
                    genres = []
                    debug_info["parsing_errors"].append(f"track.genres: {str(e)}")
                
                try:
                    lyrics_keywords_elem = track_elem.find("lyrics_keywords")
                    lyrics_keywords = parse_list_element(track_elem, "lyrics_keywords", "keyword")
                    track_field("track.lyrics_keywords", lyrics_keywords_elem, lyrics_keywords, is_list=True)
                except Exception as e:
                    lyrics_keywords = []
                    debug_info["parsing_errors"].append(f"track.lyrics_keywords: {str(e)}")
                
                try:
                    lyrics_text_elem = track_elem.find("lyrics_text")
                    lyrics_text = parse_text_or_none(lyrics_text_elem)
                    track_field("track.lyrics_text", lyrics_text_elem, lyrics_text)
                except Exception as e:
                    lyrics_text = None
                    debug_info["parsing_errors"].append(f"track.lyrics_text: {str(e)}")
                
                track = {
                    "title_keywords": title_keywords,
                    "year_from": year_from,
                    "year_to": year_to,
                    "genres": genres,
                    "lyrics_keywords": lyrics_keywords,
                    "lyrics_text": lyrics_text
                }
            else:
                track = {
                    "title_keywords": [],
                    "year_from": None,
                    "year_to": None,
                    "genres": [],
                    "lyrics_keywords": [],
                    "lyrics_text": None
                }
                debug_info["missing_fields"].append("track: element not found")
            
            # Parse artist information
            artist_elem = root.find("artist")
            artist = {}
            if artist_elem is not None:
                try:
                    name_keywords_elem = artist_elem.find("name_keywords")
                    name_keywords = parse_list_element(artist_elem, "name_keywords", "keyword")
                    track_field("artist.name_keywords", name_keywords_elem, name_keywords, is_list=True)
                except Exception as e:
                    name_keywords = []
                    debug_info["parsing_errors"].append(f"artist.name_keywords: {str(e)}")
                
                try:
                    founded_year_from_elem = artist_elem.find("founded_year_from")
                    founded_year_from = parse_int_or_none(founded_year_from_elem)
                    track_field("artist.founded_year_from", founded_year_from_elem, founded_year_from)
                except Exception as e:
                    founded_year_from = None
                    debug_info["parsing_errors"].append(f"artist.founded_year_from: {str(e)}")
                
                try:
                    founded_year_to_elem = artist_elem.find("founded_year_to")
                    founded_year_to = parse_int_or_none(founded_year_to_elem)
                    track_field("artist.founded_year_to", founded_year_to_elem, founded_year_to)
                except Exception as e:
                    founded_year_to = None
                    debug_info["parsing_errors"].append(f"artist.founded_year_to: {str(e)}")
                
                try:
                    artist_genres_elem = artist_elem.find("genres")
                    artist_genres = parse_list_element(artist_elem, "genres", "genre")
                    track_field("artist.genres", artist_genres_elem, artist_genres, is_list=True)
                except Exception as e:
                    artist_genres = []
                    debug_info["parsing_errors"].append(f"artist.genres: {str(e)}")
                
                try:
                    country_elem = artist_elem.find("country")
                    country = parse_text_or_none(country_elem)
                    track_field("artist.country", country_elem, country)
                except Exception as e:
                    country = None
                    debug_info["parsing_errors"].append(f"artist.country: {str(e)}")
                
                try:
                    region_elem = artist_elem.find("region")
                    region = parse_text_or_none(region_elem)
                    track_field("artist.region", region_elem, region)
                except Exception as e:
                    region = None
                    debug_info["parsing_errors"].append(f"artist.region: {str(e)}")
                
                try:
                    description_keywords_elem = artist_elem.find("description_keywords")
                    description_keywords = parse_list_element(artist_elem, "description_keywords", "keyword")
                    track_field("artist.description_keywords", description_keywords_elem, description_keywords, is_list=True)
                except Exception as e:
                    description_keywords = []
                    debug_info["parsing_errors"].append(f"artist.description_keywords: {str(e)}")
                
                try:
                    description_text_elem = artist_elem.find("description_text")
                    description_text = parse_text_or_none(description_text_elem)
                    track_field("artist.description_text", description_text_elem, description_text)
                except Exception as e:
                    description_text = None
                    debug_info["parsing_errors"].append(f"artist.description_text: {str(e)}")
                
                artist = {
                    "name_keywords": name_keywords,
                    "founded_year_from": founded_year_from,
                    "founded_year_to": founded_year_to,
                    "genres": artist_genres,
                    "country": country,
                    "region": region,
                    "description_keywords": description_keywords,
                    "description_text": description_text
                }
            else:
                artist = {
                    "name_keywords": [],
                    "founded_year_from": None,
                    "founded_year_to": None,
                    "genres": [],
                    "country": None,
                    "region": None,
                    "description_keywords": [],
                    "description_text": None
                }
                debug_info["missing_fields"].append("artist: element not found")
            
            # Parse features
            features_elem = root.find("features")
            features = []
            if features_elem is not None:
                try:
                    for feature in features_elem.findall("feature"):
                        if feature.text:
                            features.append(feature.text.strip())
                    track_field("features", features_elem, features, is_list=True)
                except Exception as e:
                    features = []
                    debug_info["parsing_errors"].append(f"features: {str(e)}")
            else:
                debug_info["missing_fields"].append("features: element not found in XML")
            
            # Parse album information
            album_elem = root.find("album")
            album = {}
            if album_elem is not None:
                try:
                    album_name_keywords_elem = album_elem.find("name_keywords")
                    album_name_keywords = parse_list_element(album_elem, "name_keywords", "keyword")
                    track_field("album.name_keywords", album_name_keywords_elem, album_name_keywords, is_list=True)
                    album = {
                        "name_keywords": album_name_keywords
                    }
                except Exception as e:
                    album = {
                        "name_keywords": []
                    }
                    debug_info["parsing_errors"].append(f"album.name_keywords: {str(e)}")
            else:
                album = {
                    "name_keywords": []
                }
                debug_info["missing_fields"].append("album: element not found in XML")
            
            # Print debug info
            print_debug_info(debug_info)
            
            result = {
                "track": track,
                "artist": artist,
                "features": features,
                "album": album,
                "_debug": debug_info
            }
            #json.dump(result, open("result.json", "w"), indent=4)
            return result
            
        except ET.ParseError as e:
            st.warning(f"Could not parse XML response, using fallback extraction")
            st.code(response_text)
            debug_info = {
                "parsed_fields": [],
                "missing_fields": ["XML parsing failed - ParseError"],
                "parsing_errors": [f"ET.ParseError: {str(e)}"],
                "raw_xml": response_text
            }
            print_debug_info(debug_info)
            return {
                "track": {
                    "title_keywords": [],
                    "year_from": None,
                    "year_to": None,
                    "genres": [],
                    "lyrics_keywords": [],
                    "lyrics_text": None
                },
                "artist": {
                    "name_keywords": [],
                    "founded_year_from": None,
                    "founded_year_to": None,
                    "genres": [],
                    "country": None,
                    "region": None,
                    "description_keywords": [],
                    "description_text": None
                },
                "features": [],
                "album": {
                    "name_keywords": []
                },
                "_debug": debug_info
            }
            
    except Exception as e:
        st.error(f"Error calling Claude API: {e}")
        debug_info = {
            "parsed_fields": [],
            "missing_fields": ["API call failed"],
            "parsing_errors": [f"Exception: {str(e)}"],
            "raw_xml": None
        }
        print_debug_info(debug_info)
        return {
            "track": {
                "title_keywords": [],
                "year_from": None,
                "year_to": None,
                "genres": [],
                "lyrics_keywords": [],
                "lyrics_text": None
            },
            "artist": {
                "name_keywords": [],
                "founded_year_from": None,
                "founded_year_to": None,
                "genres": [],
                "country": None,
                "region": None,
                "description_keywords": [],
                "description_text": None
            },
            "features": [],
            "album": {
                "name_keywords": []
            },
            "_debug": debug_info
        }


def static_retrieval(extracted_fields: Dict) -> List[Dict]:
    """
    Static retrieval function - returns mock song data based on extracted fields.
    In a real implementation, this would query the Neo4j GraphRAG database.
    """
    
    track_info = extracted_fields.get("track", {})
    artist_info = extracted_fields.get("artist", {})
    
    # Extract simple filters from the nested structure
    genres = track_info.get("genres", []) or artist_info.get("genres", [])
    genre = genres[0] if genres else None
    
    location = artist_info.get("country") or artist_info.get("region")
    
    artist_keywords = artist_info.get("name_keywords", [])
    artist = artist_keywords[0] if artist_keywords else None
    
    # Region to countries mapping
    region_to_countries = {
        "europe": ["United Kingdom", "France", "Germany", "Spain", "Ireland", "Croatia", "Belgium"],
        "north america": ["United States", "Canada", "Puerto Rico"],
        "south america": ["Brasil", "Argentina", "Columbia"],
        "asia": ["South Korea", "Indonesia"],
        "oceania": ["Australia", "New Zealand"]
    }
    
    # Mock song database for demonstration (with region info)
    mock_songs = [
        {
            "title": "Bohemian Rhapsody",
            "artist": "Queen",
            "genre": "rock",
            "country": "United Kingdom",
            "region": "Europe",
            "year": 1975,
            "description": "A progressive rock epic with operatic sections"
        },
        {
            "title": "Lose Yourself",
            "artist": "Eminem",
            "genre": "hip hop",
            "country": "United States",
            "region": "North America",
            "year": 2002,
            "description": "Motivational rap song about seizing opportunities"
        },
        {
            "title": "La Vie En Rose",
            "artist": "Édith Piaf",
            "genre": "chanson",
            "country": "France",
            "region": "Europe",
            "year": 1945,
            "description": "Classic French chanson about love and life"
        },
        {
            "title": "Stairway to Heaven",
            "artist": "Led Zeppelin",
            "genre": "rock",
            "country": "United Kingdom",
            "region": "Europe",
            "year": 1971,
            "description": "Epic rock ballad with acoustic and electric sections"
        },
        {
            "title": "Shape of You",
            "artist": "Ed Sheeran",
            "genre": "pop",
            "country": "United Kingdom",
            "region": "Europe",
            "year": 2017,
            "description": "Catchy pop song with tropical house influences"
        },
        {
            "title": "Gangnam Style",
            "artist": "PSY",
            "genre": "k-pop",
            "country": "South Korea",
            "region": "Asia",
            "year": 2012,
            "description": "Viral K-pop dance song"
        },
        {
            "title": "Hotel California",
            "artist": "Eagles",
            "genre": "rock",
            "country": "United States",
            "region": "North America",
            "year": 1976,
            "description": "Classic rock song with mysterious lyrics"
        },
        {
            "title": "Blinding Lights",
            "artist": "The Weeknd",
            "genre": "pop",
            "country": "Canada",
            "region": "North America",
            "year": 2019,
            "description": "Synth-pop song with 80s influences"
        },
        {
            "title": "Down Under",
            "artist": "Men at Work",
            "genre": "pop",
            "country": "Australia",
            "region": "Oceania",
            "year": 1981,
            "description": "Iconic Australian pop song"
        },
        {
            "title": "Garota de Ipanema",
            "artist": "Antônio Carlos Jobim",
            "genre": "bossa nova",
            "country": "Brasil",
            "region": "South America",
            "year": 1962,
            "description": "Classic Brazilian bossa nova"
        }
    ]
    
    # Simple filtering logic
    filtered_songs = []
    for song in mock_songs:
        match = True
        
        # Filter by genre
        if genres:
            genre_match = any(
                g.lower() in song["genre"].lower() or song["genre"].lower() in g.lower()
                for g in genres
            )
            if not genre_match:
                match = False
        
        # Filter by location (country or region)
        if location:
            location_lower = location.lower()
            # Check if location matches country
            country_match = song["country"].lower() == location_lower
            # Check if location matches region
            region_match = song["region"].lower() == location_lower
            # Check if location is a region and song's country is in that region
            region_country_match = False
            if location_lower in region_to_countries:
                region_country_match = song["country"] in region_to_countries[location_lower]
            
            if not (country_match or region_match or region_country_match):
                match = False
        
        # Filter by artist name keywords
        if artist_keywords:
            artist_match = any(
                keyword.lower() in song["artist"].lower()
                for keyword in artist_keywords
            )
            if not artist_match:
                match = False
        
        # Filter by year range
        year_from = track_info.get("year_from")
        year_to = track_info.get("year_to")
        if year_from is not None and song.get("year") is not None:
            if song["year"] < year_from:
                match = False
        if year_to is not None and song.get("year") is not None:
            if song["year"] > year_to:
                match = False
        
        # Filter by title keywords
        title_keywords = track_info.get("title_keywords", [])
        if title_keywords:
            title_match = any(
                keyword.lower() in song["title"].lower()
                for keyword in title_keywords
            )
            if not title_match:
                match = False
        
        if match:
            filtered_songs.append(song)
    
    # If no matches, return all songs (or a subset)
    if not filtered_songs:
        return mock_songs[:5]  # Return first 5 as fallback
    
    return filtered_songs[:10]  # Return up to 10 matches


def generate_song_suggestions(retrieved_songs: List[Dict], user_query: str, extracted_fields: Dict) -> str:
    """
    Use Claude to generate song suggestions based on retrieved information.
    """
    
    # Format retrieved songs for the prompt
    songs_text = "\n".join([
        f"- **{song['title']}** by {song['artist']} ({song['year']}) - {song['genre']} from {song['country']} ({song.get('region', 'N/A')}). {song['description']}"
        for song in retrieved_songs
    ])
    
    # Format extracted fields for display
    track_info = extracted_fields.get("track", {})
    artist_info = extracted_fields.get("artist", {})
    
    genres = track_info.get("genres", []) or artist_info.get("genres", [])
    genres_str = ", ".join(genres) if genres else "Not specified"
    
    location = artist_info.get("country") or artist_info.get("region") or "Not specified"
    
    artist_keywords = artist_info.get("name_keywords", [])
    artist_str = ", ".join(artist_keywords) if artist_keywords else "Not specified"
    
    year_range = ""
    if track_info.get("year_from") or track_info.get("year_to"):
        year_from = track_info.get("year_from", "?")
        year_to = track_info.get("year_to", "?")
        year_range = f"{year_from}-{year_to}"
    
    lyrics_text = track_info.get("lyrics_text") or "Not specified"
    
    suggestion_prompt = f"""You are a music recommendation assistant for MIWA (Music Indexing with AI).

The user is looking for songs based on this query: "{user_query}"

Extracted information:
- Genres: {genres_str}
- Location: {location} (country or region)
- Artist keywords: {artist_str}
- Year range: {year_range if year_range else "Not specified"}
- Lyrics theme: {lyrics_text}

Based on the user's query, I've retrieved the following songs from the database:

{songs_text}

Please analyze the user's query and the retrieved songs, then suggest the most relevant songs. 
Explain why each song matches what the user is looking for. Be conversational and helpful.

Format your response as a friendly recommendation with:
1. A brief acknowledgment of what the user is looking for
2. 3-5 most relevant song suggestions with explanations
3. Why these songs match their criteria

Keep it concise but informative."""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            temperature=0.7,
            messages=[
                {"role": "user", "content": suggestion_prompt}
            ]
        )
        
        return message.content[0].text
    except Exception as e:
        return f"Error generating suggestions: {e}"


# Chat input
if prompt := st.chat_input("Describe the song you're looking for..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show processing status
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your query..."):
            # Step 1: Extract fields
            extracted_fields = extract_fields_with_costar(prompt)
            
            # Display extracted fields in an expander
            with st.expander("🔍 Extracted Information", expanded=False):
                track_info = extracted_fields.get("track", {})
                artist_info = extracted_fields.get("artist", {})
                features_list = extracted_fields.get("features", [])
                album_info = extracted_fields.get("album", {})
                
                st.subheader("Track Information")
                col1, col2 = st.columns(2)
                with col1:
                    if track_info.get("title_keywords"):
                        st.write(f"**Title Keywords:** {', '.join(track_info['title_keywords'])}")
                    if track_info.get("genres"):
                        st.write(f"**Genres:** {', '.join(track_info['genres'])}")
                    if track_info.get("year_from") or track_info.get("year_to"):
                        year_from = track_info.get("year_from", "?")
                        year_to = track_info.get("year_to", "?")
                        st.write(f"**Year Range:** {year_from} - {year_to}")
                with col2:
                    if track_info.get("lyrics_keywords"):
                        st.write(f"**Lyrics Keywords:** {', '.join(track_info['lyrics_keywords'])}")
                    if track_info.get("lyrics_text"):
                        st.write(f"**Lyrics Theme:** {track_info['lyrics_text']}")
                
                st.subheader("Artist Information")
                col1, col2 = st.columns(2)
                with col1:
                    if artist_info.get("name_keywords"):
                        st.write(f"**Name Keywords:** {', '.join(artist_info['name_keywords'])}")
                    if artist_info.get("genres"):
                        st.write(f"**Genres:** {', '.join(artist_info['genres'])}")
                    if artist_info.get("country"):
                        st.write(f"**Country:** {artist_info['country']}")
                    if artist_info.get("region"):
                        st.write(f"**Region:** {artist_info['region']}")
                with col2:
                    if artist_info.get("founded_year_from") or artist_info.get("founded_year_to"):
                        year_from = artist_info.get("founded_year_from", "?")
                        year_to = artist_info.get("founded_year_to", "?")
                        st.write(f"**Founded Year Range:** {year_from} - {year_to}")
                    if artist_info.get("description_keywords"):
                        st.write(f"**Description Keywords:** {', '.join(artist_info['description_keywords'])}")
                    if artist_info.get("description_text"):
                        st.write(f"**Description Theme:** {artist_info['description_text']}")
                
                if features_list:
                    st.subheader("Featured Artists")
                    st.write(f"{', '.join(features_list)}")
                
                if album_info.get("name_keywords"):
                    st.subheader("Album Information")
                    st.write(f"**Album Name Keywords:** {', '.join(album_info['name_keywords'])}")
            
            # Step 2: Retrieve songs
            with st.spinner("Searching database..."):
                retrieved_songs = static_retrieval(extracted_fields)
            
            # Step 3: Generate suggestions
            with st.spinner("Generating recommendations..."):
                suggestions = generate_song_suggestions(retrieved_songs, prompt, extracted_fields)
            
            # Display suggestions
            st.markdown(suggestions)
            
            # Display retrieved songs in a table
            with st.expander("📊 Retrieved Songs from Database", expanded=False):
                if retrieved_songs:
                    import pandas as pd
                    df = pd.DataFrame(retrieved_songs)
                    display_cols = ["title", "artist", "genre", "country", "region", "year"]
                    # Only show columns that exist
                    available_cols = [col for col in display_cols if col in df.columns]
                    st.dataframe(df[available_cols], use_container_width=True)
                else:
                    st.write("No songs retrieved.")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": suggestions})

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About MIWA")
    st.markdown("""
    **MIWA** (Music Indexing with AI) helps you find songs when you don't know the exact title.
    
    **How it works:**
    1. 🔍 **Extract** - AI extracts genre, location (country/region), artist, and query details
    2. 📚 **Retrieve** - Search the music database using extracted fields
    3. 💡 **Suggest** - Get personalized song recommendations
    
    **Try queries like:**
    - "Sad rock songs from Europe"
    - "Hip hop from North America"
    - "Songs from Oceania"
    - "Pop songs from the United Kingdom"
    - "Songs by French artists"
    """)
    
    st.markdown("---")
    st.markdown("**Note:** This demo uses static retrieval. In production, it connects to a Neo4j GraphRAG database.")

