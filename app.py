import streamlit as st
import os
from anthropic import Anthropic
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
import json
from typing import Optional, Dict, List

load_dotenv()

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from collections import defaultdict
import torch
from transformers import AutoModel
import numpy as np
import json
load_dotenv()

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

model_name = "jinaai/jina-embeddings-v3"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.to(device)

uri = os.environ.get("NEO4J_URI")
username = os.environ.get("NEO4J_USERNAME")
password = os.environ.get("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))


# In[ ]:


def embed_text(text):
    if text in [None, "", []]:
        return None
    if isinstance(text, list):
        text = next((t for t in text if t not in [None, ""]), None)
        if text is None:
            return None
    emb = model.encode([text], truncate_dim=1024)
    return emb.tolist()[0]

def run_query(driver, cypher, params):
    with driver.session() as s:
        return s.run(cypher, params).data()


def list_to_or_query(items, HARD_MATCH):
    if not items:
        return None
    clean = [str(x).strip() for x in items if str(x).strip()]
    joiner = " OR " if HARD_MATCH else "~ OR "
    return " OR ".join(clean) if clean else None


# -------------------------------------------
# Atomic filters mirroring the prompt
# -------------------------------------------


def title_filter(driver, params):
    q = """
    CALL db.index.fulltext.queryNodes("track_title_fulltext", $title_keywords_query)
    YIELD node, score
    RETURN elementId(node) AS id, score
    """
    return run_query(driver, q, params)


def all_tracks_filter(driver, params):
    q = """
    MATCH (t:Track)
    RETURN elementId(t) AS id
    """
    return run_query(driver, q, params)


def track_numeric_filter(driver, params):
    q = """
    MATCH (t:Track)
    WHERE
        ($track_year_from IS NULL OR t.year >= $track_year_from)
        AND ($track_year_to IS NULL OR t.year <= $track_year_to)
        AND ($track_views_from IS NULL OR t.views >= $track_views_from)
        AND ($track_views_to IS NULL OR t.views <= $track_views_to)
    RETURN elementId(t) AS id
    """
    return run_query(driver, q, params)


def lyrics_ft_filter(driver, params):
    q = """
    CALL db.index.fulltext.queryNodes("lyrics_fulltext", $lyrics_keywords_query)
    YIELD node, score
    MATCH (t:Track)-[:HAS_LYRICS]->(node)
    RETURN elementId(t) AS id, score
    """
    return run_query(driver, q, params)

def lyrics_vec_filter(driver, params):
    q = """
    CALL db.index.vector.queryNodes("embedding_l_vector", 200, $lyrics_vector)
    YIELD node, score
    MATCH (lyr:Lyrics)-[:HAS_EMBEDDING]->(node)
    MATCH (t:Track)-[:HAS_LYRICS]->(lyr)
    RETURN elementId(t) AS id, score
    """
    return run_query(driver, q, params)

def artist_name_filter(driver, params):
    q = """
    CALL db.index.fulltext.queryNodes("artist_name_fulltext", $artist_name_keywords_query)
    YIELD node, score
    MATCH (node)<-[:PERFORMED_BY]-(t:Track)
    RETURN elementId(t) AS id, score
    """
    return run_query(driver, q, params)

def artist_description_vec_filter(driver, params):
    q = """
    CALL db.index.vector.queryNodes("embedding_d_vector", 200, $description_vector)
    YIELD node, score
    MATCH (desc:Description)-[:HAS_EMBEDDING]->(node)
    MATCH (a:Artist)-[:HAS_DESCRIPTION]->(desc)
    MATCH (a)<-[:PERFORMED_BY]-(t:Track)
    RETURN elementId(t) AS id, score
    """
    return run_query(driver, q, params)


def artist_numeric_filter(driver, params):
    q = """
    MATCH (t:Track)-[:PERFORMED_BY]->(a:Artist)
    WHERE
        ($founded_year_from IS NULL OR a.founded_year >= $founded_year_from)
        AND ($founded_year_to IS NULL OR a.founded_year <= $founded_year_to)
    RETURN elementId(t) AS id
    """
    return run_query(driver, q, params)

def location_filter(driver, params):
    q = """
    MATCH (t:Track)-[:PERFORMED_BY]->(a:Artist)-[:FOUNDED_IN]->(loc)
    WHERE
        (
            ($artist_country IS NULL OR size($artist_country) = 0 OR $artist_country = [""])
            AND ($artist_region IS NULL OR size($artist_region) = 0 OR $artist_region = [""])
        )
        OR
        (
            ($artist_region IS NOT NULL AND size($artist_region) > 0 AND $artist_region <> [""])
            AND (
                (loc:Region AND loc.name IN $artist_region)
                OR (loc:Country AND EXISTS {
                    MATCH (loc)-[:LOCATED_IN]->(r:Region)
                    WHERE r.name IN $artist_region
                })
            )
        )
        OR
        (
            ($artist_region IS NULL OR size($artist_region) = 0 OR $artist_region = [""])
            AND ($artist_country IS NOT NULL AND size($artist_country) > 0 AND $artist_country <> [""])
            AND loc:Country AND loc.name IN $artist_country
        )
    RETURN elementId(t) AS id
    """
    return run_query(driver, q, params)


def artist_genre_filter(driver, params):
    # q = """
    # UNWIND $artist_genre_keywords AS kw
    # CALL {
    #     WITH kw
    #     CALL db.index.fulltext.queryNodes("genre_fulltext", kw) YIELD node, score
    #     RETURN node ORDER BY score DESC LIMIT 1
    # }
    # WITH collect(distinct node) AS top_nodes
    # UNWIND top_nodes AS node
    # MATCH (node)<-[:IS_SUBGENRE*0..]-(sub)
    # WITH collect(distinct sub) AS allowed
    # MATCH (t:Track)-[:PERFORMED_BY]->(a:Artist)-[:PERFORMS_GENRE]->(g:Genre)
    # WHERE g IN allowed
    # RETURN elementId(t) AS id
    # """
    q = """
    UNWIND $artist_genre_keywords AS kw
    CALL (kw) {
        CALL db.index.fulltext.queryNodes("genre_fulltext", kw) YIELD node, score
        RETURN node ORDER BY score DESC LIMIT 1
    }
    WITH collect(distinct node) AS top_nodes
    UNWIND top_nodes AS node
    MATCH (node)<-[:IS_SUBGENRE*0..]-(sub)
    WITH collect(distinct sub) AS allowed
    MATCH (t:Track)-[:PERFORMED_BY]->(a:Artist)-[:PERFORMS_GENRE]->(g:Genre)
    WHERE g IN allowed
    RETURN elementId(t) AS id
    """


    return run_query(driver, q, params)


def track_genre_filter(driver, params):
    # q = """
    # UNWIND $track_genre_keywords AS kw
    # CALL {
    #     WITH kw
    #     CALL db.index.fulltext.queryNodes("genre_fulltext", kw) YIELD node, score
    #     RETURN node ORDER BY score DESC LIMIT 1
    # }
    # WITH collect(distinct node) AS top_nodes
    # UNWIND top_nodes AS node
    # MATCH (node)<-[:IS_SUBGENRE*0..]-(sub)
    # WITH collect(distinct sub) AS allowed
    # MATCH (t:Track)-[:HAS_GENRE]->(g:Genre)
    # WHERE g IN allowed
    # RETURN elementId(t) AS id
    # """
    q = """
    UNWIND $track_genre_keywords AS kw
    CALL (kw) {
        CALL db.index.fulltext.queryNodes("genre_fulltext", kw) YIELD node, score
        RETURN node ORDER BY score DESC LIMIT 1
    }
    WITH collect(distinct node) AS top_nodes
    UNWIND top_nodes AS node
    MATCH (node)<-[:IS_SUBGENRE*0..]-(sub)
    WITH collect(distinct sub) AS allowed
    MATCH (t:Track)-[:HAS_GENRE]->(g:Genre)
    WHERE g IN allowed
    RETURN elementId(t) AS id
    """
    return run_query(driver, q, params)


def features_filter(driver, params):
    q = """
    UNWIND $features AS kw
    CALL {
        WITH kw
        CALL db.index.fulltext.queryNodes("artist_name_fulltext", kw) YIELD node, score
        RETURN node ORDER BY score DESC LIMIT 10
    }
    WITH collect(distinct node) AS allowed_feature_artists
    MATCH (t:Track)-[:FEATURING]->(feat:Artist)
    WHERE feat IN allowed_feature_artists
    RETURN elementId(t) AS id
    """
    return run_query(driver, q, params)

# def album_filter(driver, params):
#     q = """
#     MATCH (t:Track)-[:APPEARS_ON]->(alb:Album)
#     WHERE
#         ($album_views_from IS NULL OR alb.playcount >= $album_views_from)
#         AND ($album_views_to IS NULL OR alb.playcount <= $album_views_to)
#         AND (
#             $album_name_keywords IS NULL OR size($album_name_keywords) = 0 OR $album_name_keywords = [""]
#             OR EXISTS {
#                 UNWIND $album_name_keywords AS kw
#                 CALL db.index.fulltext.queryNodes("album_name_fulltext", kw) YIELD node, score
#                 WITH node, score ORDER BY score DESC LIMIT 1
#                 WHERE node = alb
#             }
#         )
#     RETURN elementId(t) AS id
#     """
#     return run_query(driver, q, params)
def album_filter(driver, params):
    q = """
    CALL {
        WITH $album_name_keywords AS kws
        WHERE kws IS NOT NULL AND size(kws) > 0 AND kws[0] <> ""
        
        WITH apoc.text.join($album_name_keywords, " OR ") AS queryStr
        CALL db.index.fulltext.queryNodes("album_name_fulltext", queryStr) YIELD node, score
        RETURN node AS alb, score
        
        UNION
        
        WITH $album_name_keywords AS kws
        WHERE kws IS NULL OR size(kws) = 0 OR kws[0] = ""
        MATCH (alb:Album)
        RETURN alb, 0.0 AS score
    }

    WITH alb, score
    WHERE ($album_views_from IS NULL OR alb.playcount >= $album_views_from)
    AND ($album_views_to IS NULL OR alb.playcount <= $album_views_to)

    ORDER BY score DESC, alb.playcount DESC
    LIMIT 50

    MATCH (t:Track)-[:APPEARS_ON]->(alb)
    RETURN 
        elementId(t) AS id, 
        score 
    """
    return run_query(driver, q, params)

# -------------------------------------------
# Main filtering logic
# -------------------------------------------


def filter_tracks_with_scoring(driver, p):
    score_table = defaultdict(lambda: defaultdict(float))
    candidate_ids = None

    # Ordered by likely selectivity
    filter_sequence = [("filter_track_numeric", track_numeric_filter)]

    if p.get("title_keywords_query"):
        filter_sequence.append(("score_title", title_filter))

    if p.get("lyrics_keywords_query"):
        filter_sequence.append(("score_lyrics_ft", lyrics_ft_filter))

    if p.get("lyrics_vector"):
        # print("lyrics")
        filter_sequence.append(("score_lyrics_vec", lyrics_vec_filter))

    if p.get("artist_name_keywords_query"):
        filter_sequence.append(("score_artist_name", artist_name_filter))
    # print(p.get("description_vector"))
    # print(p.get("lyrics_vector"))
    if p.get("description_vector"):
        # print("desc")
        filter_sequence.append(("score_artist_desc", artist_description_vec_filter))

    if p.get("track_genre_keywords"):
        filter_sequence.append(("filter_track_genre", track_genre_filter))

    if p.get("artist_genre_keywords"):
        filter_sequence.append(("filter_artist_genre", artist_genre_filter))

    if p.get("features"):
        filter_sequence.append(("filter_features", features_filter))

    if p.get("artist_country") or p.get("artist_region"):
        filter_sequence.append(("filter_location", location_filter))

    if p.get("founded_year_from") or p.get("founded_year_to"):
        filter_sequence.append(("filter_artist_numeric", artist_numeric_filter))

    if p.get("album_name_keywords") or p.get("album_views_from") or p.get("album_views_to"):
        filter_sequence.append(("score_filter_album", album_filter))

    if not filter_sequence:
        filter_sequence.append(("filter_all", all_tracks_filter))

    # -------------------------------
    # Execute filters one by one
    # -------------------------------
    for score_key, func in filter_sequence:

        rows = func(driver, p)
        ids = [r["id"] for r in rows]
        # print(score_key)
        # print(ids)

        # intersection if previous exists
        if candidate_ids is not None:
            ids = list(set(candidate_ids) & set(ids))
        candidate_ids = ids

        # assign scores only if this is scoring filter
        if score_key.startswith("score_"):
            for r in rows:
                if r["id"] in candidate_ids:
                    score_table[r["id"]][score_key] = r.get("score", 0.0)

        # early stop
        # if not candidate_ids:
        #     return []

    # -------------------------------
    # Fetch final track names
    # -------------------------------

    if not candidate_ids:
        return []

    name_query = """
    MATCH (t:Track)
    WHERE elementId(t) IN $ids

    // artist(s)
    OPTIONAL MATCH (t)-[:PERFORMED_BY]->(artist:Artist)
    OPTIONAL MATCH (t)-[:FEATURING]->(feat:Artist)

    // album
    OPTIONAL MATCH (t)-[:APPEARS_ON]->(al:Album)

    // genres
    OPTIONAL MATCH (t)-[:HAS_GENRE]->(g:Genre)

    // lyrics + embeddings (if needed)
    OPTIONAL MATCH (t)-[:HAS_LYRICS]->(ly:Lyrics)

    // artist → country → region → planet
    OPTIONAL MATCH (artist)-[:FOUNDED_IN]->(c:Country)
    OPTIONAL MATCH (c)-[:LOCATED_IN]->(r:Region)
    OPTIONAL MATCH (artist)-[:PERFORMS_GENRE]->(ag:Genre)

    // description
    OPTIONAL MATCH (artist)-[:HAS_DESCRIPTION]->(desc:Description)

    RETURN
        elementId(t) AS track_id,
        t.title AS track_title,
        t.views AS track_views,
        t.year AS track_year,

        artist.name AS artist_name,
        artist.founded_year AS artist_founded_year,

        al.name AS album_name,
        al.playcount AS album_playcount,

        collect(DISTINCT g.name) AS genres,
        collect(DISTINCT ag.name) AS artist_genres,
        collect(DISTINCT feat.name) AS featuring_artists,

        c.name AS country,
        r.name AS region,

        ly.text AS lyrics,
        desc.text AS description

    """
    
    names = run_query(driver, name_query, {"ids": candidate_ids})
    print("Total retrieved: ", len(names))
    # name_map = {r["track_id"]: r["track_title"] for r in names}
    name_map = {r["track_id"]: r for r in names}
    
    results = []
    for tid in candidate_ids:
        scores = score_table[tid]
        # Use max of full-text or vector lyrics scores
        # lyrics_score = max(scores.get("score_lyrics_ft", 0.0), scores.get("score_lyrics_vec", 0.0))
        
        total = (
            scores.get("score_title", 0.0)
            + scores.get("score_lyrics_ft", 0.0)
            + scores.get("score_lyrics_vec", 0.0)
            + scores.get("score_artist_name", 0.0)
            + scores.get("score_artist_desc", 0.0)
            + scores.get("score_filter_album", 0.0)
        )
        
        track_info = name_map.get(tid)
        if track_info and isinstance(track_info, dict):
            results.append({
                "track_id": tid,
                "score_total": total,
                "track_title": track_info.get("track_title", "Unknown"),
                "artist_name": track_info.get("artist_name", "Unknown"),
                "genres": track_info.get("genres", []),
                "country": track_info.get("country"),
                "region": track_info.get("region"),
                "track_year": track_info.get("track_year"),
            })
        else:
            # Fallback if track info not found
            results.append({
                "track_id": tid,
                "score_total": total,
                "track_title": "Unknown",
                "artist_name": "Unknown",
                "genres": [],
                "country": None,
                "region": None,
                "track_year": None,
            })

    # 2. Sort by total score descending
    results.sort(key=lambda x: x["score_total"], reverse=True)
    top15 = results[:15]

    # 3. Print the detailed records for the top 15
    # print(f"\n--- Top {len(top15)} Search Results ---")
    # for item in top15:
    #     tid = item["track_id"]
    #     full_record = name_map.get(tid)
        
    #     if full_record:
    #         # Add the calculated score to the record so the printer can show it if needed
    #         full_record["score_total"] = item["score_total"]
    #         pretty_print_full(full_record)
    #     else:
    #         print(f"ID {tid}: Detailed data not found. Score: {item['score_total']}")

    return top15



def pretty_print_full(result):
    print("\n================ TRACK ==================")
    print(f"🎯 Score: {result['score_total']}")
    print(f"🎵 Title: {result['track_title']}")
    print(f"📅 Year: {result.get('track_year', 'N/A')}")
    print(f"🏷️ Genre(s): {', '.join(result.get('genres', []))}")
    print(f"👁️ Views: {result.get('track_views', 'N/A')}")
    feats = None
    if result.get("featuring_artists"):
        feats = " 🤝 Feat. " + ", ".join(result["featuring_artists"])
    print(f"👤 Artist: {result['artist_name']}{feats if feats else ''}")
    print(f"🏷️ Genres: {', '.join(result['artist_genres'])}")
    print(f"🌎 Country: {result.get('country', 'N/A')}")
    print(f"🗺️ Region: {result.get('region', 'N/A')}")

    print(f"💿 Album: {result.get('album_name', 'N/A')}")
    print(f"🔥 Album Playcount: {result.get('album_playcount', 'N/A')}")


# In[46]:


def parse_parameters(input_json, HARD_MATCH=False):
    track = input_json.get("track", {}) or {}
    artist = input_json.get("artist", {}) or {}

    album = input_json.get("album", {}) or {}
    features = input_json.get("features", []) or []
    # print(artist)
    # print(track)
    # print(album)
    # print(features)
    
    lyrics_emb = embed_text(track.get("lyrics_text")) if track.get("lyrics_text") else []
    desc_emb = embed_text(artist.get("description_text")) if artist.get("description_text") else []
    params = {
        "metadata_limit": input_json.get("metadata_limit", 15),

        # track
        "track_year_from": track.get("year_from"),
        "track_year_to": track.get("year_to"),
        "track_views_from": track.get("views_from")[0] if isinstance(track.get("views_from"), list) else track.get("views_from"),
        "track_views_to": track.get("views_to")[0] if isinstance(track.get("views_to"), list) else track.get("views_to"),
        "track_genre_keywords": track.get("genres") or [],
        "title_keywords_query": list_to_or_query(track.get("title_keywords"), HARD_MATCH),
        "lyrics_keywords_query": list_to_or_query(track.get("lyrics_keywords"), HARD_MATCH),
        "lyrics_vector": lyrics_emb,

        # artist
        "artist_name_keywords_query": list_to_or_query(artist.get("name_keywords"), HARD_MATCH),
        "founded_year_from": artist.get("founded_year_from"),
        "founded_year_to": artist.get("founded_year_to"),
        "artist_genre_keywords": artist.get("genres") or [],
        "artist_country": [] if artist.get("country", "") == "" else [artist.get("country")],
        "artist_region": [] if artist.get("region", None) is None else [],
        "description_vector": desc_emb,

        # features
        "features": features or [],

        # album
        "album_name_keywords": album.get("name_keywords") or [],
        "album_views_from": album.get("views_from"),
        "album_views_to": album.get("views_to"),
    }

    return params

def search_neo4j(input_json, HARD_MATCH=False):
    driver = GraphDatabase.driver(uri, auth=(username, password))

    params = parse_parameters(input_json, HARD_MATCH)
    results = filter_tracks_with_scoring(driver, params)
    return results
        

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
    Only prints if debug mode is enabled in session state.
    """
    # Check if debug mode is enabled
    if not st.session_state.get("show_debug_info", False):
        return
    
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
- views_from and views_to: Ignore these fields - do not extract them from the query

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
- views_from and views_to: Ignore these fields - do not extract them from the query

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
                    "views_from": None,
                    "views_to": None,
                    "lyrics_keywords": lyrics_keywords,
                    "lyrics_text": lyrics_text
                }
            else:
                track = {
                    "title_keywords": [],
                    "year_from": None,
                    "year_to": None,
                    "genres": [],
                    "views_from": None,
                    "views_to": None,
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
                    country_text = parse_text_or_none(country_elem)
                    # Use empty string instead of None for country
                    country = country_text if country_text is not None else ""
                    track_field("artist.country", country_elem, country)
                except Exception as e:
                    country = ""
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
                    description_text_value = parse_text_or_none(description_text_elem)
                    # Use empty string instead of None for description_text
                    description_text = description_text_value if description_text_value is not None else ""
                    track_field("artist.description_text", description_text_elem, description_text)
                except Exception as e:
                    description_text = ""
                    debug_info["parsing_errors"].append(f"artist.description_text: {str(e)}")
                
                artist = {
                    "name_keywords": name_keywords,
                    "founded_year_from": founded_year_from,
                    "founded_year_to": founded_year_to,
                    "genres": artist_genres,
                    "country": country if country is not None else "",
                    "region": region,
                    "description_keywords": description_keywords,
                    "description_text": description_text if description_text is not None else ""
                }
            else:
                artist = {
                    "name_keywords": [],
                    "founded_year_from": None,
                    "founded_year_to": None,
                    "genres": [],
                    "country": "",
                    "region": None,
                    "description_keywords": [],
                    "description_text": ""
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
                        "name_keywords": album_name_keywords,
                        "views_from": [],
                        "views_to": []
                    }
                except Exception as e:
                    album = {
                        "name_keywords": [],
                        "views_from": [],
                        "views_to": []
                    }
                    debug_info["parsing_errors"].append(f"album.name_keywords: {str(e)}")
            else:
                album = {
                    "name_keywords": [],
                    "views_from": [],
                    "views_to": []
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
            json.dump(result, open("result.json", "w"), indent=4)
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
                    "views_from": None,
                    "views_to": None,
                    "lyrics_keywords": [],
                    "lyrics_text": None
                },
                "artist": {
                    "name_keywords": [],
                    "founded_year_from": None,
                    "founded_year_to": None,
                    "genres": [],
                    "country": "",
                    "region": None,
                    "description_keywords": [],
                    "description_text": ""
                },
                "features": [],
                "album": {
                    "name_keywords": [],
                    "views_from": [],
                    "views_to": []
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
                "views_from": None,
                "views_to": None,
                "lyrics_keywords": [],
                "lyrics_text": None
            },
            "artist": {
                "name_keywords": [],
                "founded_year_from": None,
                "founded_year_to": None,
                "genres": [],
                "country": "",
                "region": None,
                "description_keywords": [],
                "description_text": ""
            },
            "features": [],
            "album": {
                "name_keywords": [],
                "views_from": [],
                "views_to": []
            },
            "_debug": debug_info
        }


def retrieve_songs(extracted_fields: Dict) -> List[Dict]:
    """
    Retrieve songs from Neo4j GraphRAG database using extracted fields.
    """
    # Remove debug info if present
    query_json = {k: v for k, v in extracted_fields.items() if k != "_debug"}
    
    try:
        # Initialize Neo4j retriever
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not all([uri, username, password]):
            st.error("Neo4j credentials not configured. Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in your environment variables.")
            return []
        
        results = search_neo4j(query_json, HARD_MATCH=False)
        
        # Format results for display
        formatted_songs = []
        for result in results:
            song_data = {
                "title": result.get("track_title", "Unknown"),
                "artist": result.get("artist_name", "Unknown"),
                "genre": ", ".join(result.get("genres", [])) if result.get("genres") else "Unknown",
                "country": result.get("country", ""),
                "region": result.get("region", ""),
                "year": result.get("track_year"),
                "_score": result.get("score_total", 0)
            }
            formatted_songs.append(song_data)
        
        return formatted_songs
            
    except Exception as e:
        st.error(f"Error retrieving songs from Neo4j: {e}")
        return []


def generate_song_suggestions(retrieved_songs: List[Dict], user_query: str, extracted_fields: Dict) -> str:
    """
    Use Claude to generate song suggestions based on retrieved information.
    """
    
    # Format retrieved songs for the prompt
    songs_text = "\n".join([
        f"- **{song['title']}**"
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
                retrieved_songs = retrieve_songs(extracted_fields)
            
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
                    display_cols = ["title", "artist", "genre", "country", "region", "year", "_score"]
                    # Only show columns that exist
                    available_cols = [col for col in display_cols if col in df.columns]
                    # Rename _score for display
                    if "_score" in available_cols:
                        df = df.rename(columns={"_score": "relevance_score"})
                        available_cols = [col if col != "_score" else "relevance_score" for col in available_cols]
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
    
    # Debug mode toggle
    if "show_debug_info" not in st.session_state:
        st.session_state.show_debug_info = False
    
    st.checkbox(
        "🐛 Show Debug Information",
        value=st.session_state.show_debug_info,
        key="show_debug_info",
        help="Enable to see detailed extraction debug information (parsed fields, errors, raw XML)"
    )
    
    st.markdown("---")
    st.markdown("**Note:** This demo connects to a Neo4j GraphRAG database for song retrieval.")

