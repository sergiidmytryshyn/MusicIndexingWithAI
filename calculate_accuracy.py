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
            "score_total": total,
            "track_title": name_map.get(tid).get("track_title"),
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


def match_track_names(name1: str, name2: str) -> bool:
    """
    Check if two track names match (non-strict).
    Returns True if name1 is in name2 or name2 is in name1 (case-insensitive).
    """
    if not name1 or not name2:
        return False
    
    name1_clean = name1.lower().strip()
    name2_clean = name2.lower().strip()
    
    # Exact match
    if name1_clean == name2_clean:
        return True
    
    # Check if one contains the other
    if name1_clean in name2_clean or name2_clean in name1_clean:
        return True
    
    return False


def calculate_accuracy(prompts_file: str = "data_parsing/data/generated_prompts.json"):
    """
    Calculate retrieval accuracy metrics (top-1, top-3, top-10, top-20).
    """
    print(f"Loading prompts from: {prompts_file}")
    
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File {prompts_file} not found!")
        return None
    
    prompts_list = data.get('prompts', [])[:100]
    print(f"Found {len(prompts_list)} prompts\n")
    
    if not prompts_list:
        print("❌ No prompts found in file!")
        return None
    
    # Initialize counters
    total_tests = 0
    top1_correct = 0
    top3_correct = 0
    top10_correct = 0
    top15_correct = 0
    
    results = []
    
    print("=" * 80)
    print("Calculating accuracy metrics...")
    print("=" * 80)
    
    for i, entry in enumerate(prompts_list, 1):
        track_id = entry.get('track_id')
        expected_title = entry.get('title', '')
        expected_artist = entry.get('artist_name', '')
        extracted_json = entry.get('extracted_json', {})
        
        if not extracted_json:
            print(f"[{i}/{len(prompts_list)}] ⚠️  No extracted_json found, skipping...")
            continue
        
        if not expected_title:
            print(f"[{i}/{len(prompts_list)}] ⚠️  No expected title found, skipping...")
            continue
        
        print(f"\n[{i}/{len(prompts_list)}] Testing: {expected_title} - {expected_artist}")
        
        try:
            # Retrieve songs using extracted JSON
            retrieved_results = search_neo4j(extracted_json, HARD_MATCH=False)
            
            if not retrieved_results:
                print(f"  ⚠️  No results retrieved")
                results.append({
                    "track_id": track_id,
                    "expected_title": expected_title,
                    "expected_artist": expected_artist,
                    "found": False,
                    "top1": False,
                    "top3": False,
                    "top10": False,
                    "top15": False,
                    "retrieved_count": 0
                })
                total_tests += 1
                continue
            
            # Extract track titles from retrieved results
            retrieved_titles = []
            for result in retrieved_results:
                #track = result.get('track', {})
                retrieved_title = result.get('track_title', '')
                if retrieved_title:
                    retrieved_titles.append(retrieved_title)
            
            print(f"  Retrieved {len(retrieved_titles)} tracks")
            
            # Check matches at different top-K levels
            top1_match = False
            top3_match = False
            top10_match = False
            top15_match = False
            
            # Check top-1
            if len(retrieved_titles) > 0:
                if match_track_names(expected_title, retrieved_titles[0]):
                    top1_match = True
                    top1_correct += 1
            
            # Check top-3
            if len(retrieved_titles) >= 3:
                top3_titles = retrieved_titles[:3]
            else:
                top3_titles = retrieved_titles
            
            if any(match_track_names(expected_title, title) for title in top3_titles):
                top3_match = True
                top3_correct += 1
            
            # Check top-10
            if len(retrieved_titles) >= 10:
                top10_titles = retrieved_titles[:10]
            else:
                top10_titles = retrieved_titles
            
            if any(match_track_names(expected_title, title) for title in top10_titles):
                top10_match = True
                top10_correct += 1
            
            # Check top-15
            if len(retrieved_titles) >= 15:
                top15_titles = retrieved_titles[:15]
            else:
                top15_titles = retrieved_titles
            
            if any(match_track_names(expected_title, title) for title in top15_titles):
                top15_match = True
                top15_correct += 1
            
            # Print match status
            match_status = []
            if top1_match:
                match_status.append("✅ Top-1")
            if top3_match:
                match_status.append("✅ Top-3")
            if top10_match:
                match_status.append("✅ Top-10")
            if top15_correct:
                match_status.append("✅ Top-15")
            
            if match_status:
                print(f"  {' | '.join(match_status)}")
            else:
                print(f"  ❌ No match found")
                if retrieved_titles:
                    print(f"     Top retrieved: {retrieved_titles[0]}")
            
            results.append({
                "track_id": track_id,
                "expected_title": expected_title,
                "expected_artist": expected_artist,
                "found": len(retrieved_titles) > 0,
                "top1": top1_match,
                "top3": top3_match,
                "top10": top10_match,
                "top15": top15_correct,
                "retrieved_count": len(retrieved_titles),
                "top_retrieved_titles": retrieved_titles[:5]  # Store top 5 for debugging
            })
            
            total_tests += 1
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results.append({
                "track_id": track_id,
                "expected_title": expected_title,
                "expected_artist": expected_artist,
                "error": str(e),
                "found": False,
                "top1": False,
                "top3": False,
                "top10": False,
                "top15": False,
                "retrieved_count": 0
            })
            total_tests += 1
    
    # Calculate accuracy metrics
    if total_tests == 0:
        print("\n❌ No tests completed!")
        return None
    
    top1_accuracy = (top1_correct / total_tests) * 100
    top3_accuracy = (top3_correct / total_tests) * 100
    top10_accuracy = (top10_correct / total_tests) * 100
    top15_accuracy = (top15_correct / total_tests) * 100
    
    # Print summary
    print("\n" + "=" * 80)
    print("ACCURACY SUMMARY")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"\nTop-1 Accuracy:  {top1_correct}/{total_tests} = {top1_accuracy:.2f}%")
    print(f"Top-3 Accuracy:  {top3_correct}/{total_tests} = {top3_accuracy:.2f}%")
    print(f"Top-10 Accuracy: {top10_correct}/{total_tests} = {top10_accuracy:.2f}%")
    print(f"Top-15 Accuracy: {top15_correct}/{total_tests} = {top15_accuracy:.2f}%")
    print("=" * 80)
    
    # Save detailed results
    output_data = {
        "summary": {
            "total_tests": total_tests,
            "top1_accuracy": top1_accuracy,
            "top3_accuracy": top3_accuracy,
            "top10_accuracy": top10_accuracy,
            "top15_accuracy": top15_accuracy,
            "top1_correct": top1_correct,
            "top3_correct": top3_correct,
            "top10_correct": top10_correct,
            "top15_correct": top15_correct
        },
        "detailed_results": results
    }
    
    output_file = "accuracy_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detailed results saved to: {output_file}")
    
    return output_data


if __name__ == "__main__":
    calculate_accuracy()


