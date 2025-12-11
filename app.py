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

def extract_fields_with_costar(user_query: str) -> Dict[str, Optional[str]]:
    """
    Extract genre, location (country/region), artist, and clean query from user input using COSTAR format.
    Returns a dictionary with extracted fields.
    """
    
    costar_prompt = f"""<CONTEXT>
You are a music information extraction system. Your task is to extract structured information from user queries about songs they want to find.

The user may mention:
- Genre (e.g., "rock", "pop", "hip hop", "jazz")
- Location: Either a country (e.g., "United States", "France", "UK") OR a region (e.g., "Europe", "North America", "Oceania", "Asia", "South America")
- Artist name (e.g., "The Beatles", "Eminem")
- Mood or lyrics snippets (e.g., "sad songs", "songs about love", "I remember when...")

Extract only the information that is explicitly mentioned or can be clearly inferred. If a field is not mentioned, return null for that field.
</CONTEXT>

<OBJECTIVE>
Extract the following fields from the user query:
1. genre: The musical genre mentioned (if any)
2. location: The location mentioned - can be either a country (e.g., "United States", "France", "United Kingdom") OR a region (e.g., "Europe", "North America", "Oceania", "Asia", "South America")
3. artist: The artist name mentioned (if any)
4. clean_query: The cleaned query text, removing any redundant information that was already extracted into the above fields. Keep mood, lyrics snippets, and other descriptive elements.

Output ONLY valid XML with no additional text, comments, or explanations.
</OBJECTIVE>

<STYLE>
- Be precise and only extract what is clearly mentioned
- For genre, use lowercase (e.g., "rock", "hip hop", "r&b")
- For location, use standard names: countries like "United States", "United Kingdom", "France", "Canada", "Australia" OR regions like "Europe", "North America", "South America", "Asia", "Oceania"
- For artist, use the exact name as mentioned
- For clean_query, preserve the user's intent but remove redundant extracted information
</STYLE>

<AUDIENCE>
Music search system that will use these fields for retrieval
</AUDIENCE>

<RESPONSE_FORMAT>
Output ONLY XML in the following format, with no other text:

<extraction>
    <genre>genre_value_or_null</genre>
    <location>location_value_or_null</location>
    <artist>artist_value_or_null</artist>
    <clean_query>cleaned_query_text</clean_query>
</extraction>
</RESPONSE_FORMAT>

<REASONING>
Think step by step:
1. Identify if genre is mentioned
2. Identify if location (country OR region) is mentioned - check for both countries (e.g., "United States", "France") and regions (e.g., "Europe", "North America", "Oceania")
3. Identify if artist is mentioned
4. Create clean_query by removing redundant extracted info but keeping mood/lyrics/descriptions
</REASONING>

User Query: {user_query}

Now output ONLY the XML response with no additional text:"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
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
            
            root = ET.fromstring(xml_text)
            # Support both "country" (legacy) and "location" (new) for backward compatibility
            location_elem = root.find("location") if root.find("location") is not None else root.find("country")
            result = {
                "genre": root.find("genre").text if root.find("genre") is not None and root.find("genre").text else None,
                "location": location_elem.text if location_elem is not None and location_elem.text else None,
                "artist": root.find("artist").text if root.find("artist") is not None and root.find("artist").text else None,
                "clean_query": root.find("clean_query").text if root.find("clean_query") is not None and root.find("clean_query").text else None
            }
            
            # Clean up None values and "null" strings
            for key in result:
                if result[key] and (result[key].lower() == "null" or result[key].strip() == ""):
                    result[key] = None
                elif result[key]:
                    result[key] = result[key].strip()
            st.write(result)
            return result
        except ET.ParseError as e:
            st.warning(f"Could not parse XML response, using fallback extraction")
            st.code(response_text)
            return {
                "genre": None,
                "location": None,
                "artist": None,
                "clean_query": user_query
            }
            
    except Exception as e:
        st.error(f"Error calling Claude API: {e}")
        return {
            "genre": None,
            "location": None,
            "artist": None,
            "clean_query": user_query
        }


def static_retrieval(genre: Optional[str], location: Optional[str], artist: Optional[str], clean_query: Optional[str]) -> List[Dict]:
    """
    Static retrieval function - returns mock song data based on extracted fields.
    In a real implementation, this would query the Neo4j GraphRAG database.
    """
    
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
        
        if genre and song["genre"].lower() != genre.lower():
            # Check if genre is similar (simple matching)
            if genre.lower() not in song["genre"].lower() and song["genre"].lower() not in genre.lower():
                match = False
        
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
        
        if artist and song["artist"].lower() != artist.lower():
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
    
    suggestion_prompt = f"""You are a music recommendation assistant for MIWA (Music Indexing with AI).

The user is looking for songs based on this query: "{user_query}"

Extracted information:
- Genre: {extracted_fields.get('genre', 'Not specified')}
- Location: {extracted_fields.get('location', 'Not specified')} (can be country or region)
- Artist: {extracted_fields.get('artist', 'Not specified')}
- Clean Query: {extracted_fields.get('clean_query', user_query)}

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
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Genre:** {extracted_fields.get('genre', 'Not specified')}")
                    st.write(f"**Location:** {extracted_fields.get('location', 'Not specified')} (country/region)")
                with col2:
                    st.write(f"**Artist:** {extracted_fields.get('artist', 'Not specified')}")
                    st.write(f"**Clean Query:** {extracted_fields.get('clean_query', prompt)}")
            
            # Step 2: Retrieve songs
            with st.spinner("Searching database..."):
                retrieved_songs = static_retrieval(
                    genre=extracted_fields.get("genre"),
                    location=extracted_fields.get("location"),
                    artist=extracted_fields.get("artist"),
                    clean_query=extracted_fields.get("clean_query")
                )
            
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

