#!/usr/bin/env python
# coding: utf-8

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from collections import defaultdict
import torch
from transformers import AutoModel
import numpy as np

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
    q = """
    UNWIND $artist_genre_keywords AS kw
    CALL {
        WITH kw
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
        
        results.append({
            "track_id": tid,
            "score_total": total
        })

    # 2. Sort by total score descending
    results.sort(key=lambda x: x["score_total"], reverse=True)
    top15 = results[:15]

    # 3. Print the detailed records for the top 15
    print(f"\n--- Top {len(top15)} Search Results ---")
    for item in top15:
        tid = item["track_id"]
        full_record = name_map.get(tid)
        
        if full_record:
            # Add the calculated score to the record so the printer can show it if needed
            full_record["score_total"] = item["score_total"]
            pretty_print_full(full_record)
        else:
            print(f"ID {tid}: Detailed data not found. Score: {item['score_total']}")

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
        "artist_country": artist.get("country") or [],
        "artist_region": artist.get("region") or [],
        "description_vector": desc_emb,

        # features
        "features": features or [],

        # album
        "album_name_keywords": album.get("name_keywords") or [],
        "album_views_from": album.get("views_from"),
        "album_views_to": album.get("views_to"),
    }

    return params

def miwa(input_json, HARD_MATCH=False):
    driver = GraphDatabase.driver(uri, auth=(username, password))

    params = parse_parameters(input_json, HARD_MATCH)
    results = filter_tracks_with_scoring(driver, params)


extracted_json =  {
    "track": {
        # "title_keywords": ["father", "son"],
        # "title_keywords": ["dunkelheit"],
        "title_keywords": [],
        "year_from": 1970,
        "year_to": 1999,
        "genres": ["rock", "metal"],
        "views_from": [100],
        "views_to": [10000000],
        # "lyrics_keywords": ["lightning", "thunder"],
        "lyrics_keywords": [],
        # "lyrics_text": "from father to son"
        "lyrics_text": ""
    },
    "artist": {
        "name_keywords": [],
        "founded_year_from": 1,
        "founded_year_to": 2299,
        "genres": ["metal"],
        # "country": ["Sweden"],
        "country": [],
        # "region": ["Europe"],
        "region": [],
        "description_keywords": [],
        "description_text": "Scandinavian black metal band named bathory"
    },
    "features": [],
    "album": {
        "name_keywords": ["fire"],
        "views_from": None,
        "views_to": None
    }
}

miwa(extracted_json, HARD_MATCH=0)


