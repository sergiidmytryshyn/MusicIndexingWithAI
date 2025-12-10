import os
import re
import time
import json
from neo4j import GraphDatabase
from typing import List, Optional
from dotenv import load_dotenv


load_dotenv()


class GraphCreator:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.driver.verify_connectivity()
        print("Connection established successfully!")

        self.embedding_lyric_dim = 1024
        self.embedding_description_dim = 1024

    def close(self):
        self.driver.close()

    @staticmethod
    def read_json(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def parse_features_string(feature_str):
        clean_str = feature_str.strip("{}")
        if not clean_str:
            return []
        clean_str = re.sub(r"\\+", "", clean_str)        
        clean_str = clean_str.replace('"', '')

        final_list = clean_str.split(",")
        return final_list
    

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap_ratio: float = 0.2) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []

        overlap = int(chunk_size * overlap_ratio)
        step = max(1, chunk_size - overlap)

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            chunk = text[start:start + chunk_size]
            if not chunk:
                break
            chunks.append(chunk)
            if len(chunk) < chunk_size:
                break
            start += step

        return chunks

    @staticmethod
    def _process_subgenres(tx, child_genre, parent_name):
        """
        Recursively processes a genre and its subgenres, creating nodes and
        the :IS_SUBGENRE relationship pointing to the parent.
        """
        child_name = child_genre["name"]
        
        # 1. MERGE the child Genre node
        tx.run("MERGE (:Genre {name: $name})", name=child_name)
        
        # 2. MERGE the IS_SUBGENRE relationship (child -> parent)
        query_link = """
        MATCH (p:Genre {name: $parent_name})
        MATCH (c:Genre {name: $child_name})
        MERGE (c)-[:IS_SUBGENRE]->(p)
        """
        tx.run(query_link, parent_name=parent_name, child_name=child_name)
        
        # 3. Recurse for deeper levels
        for subgenre in child_genre.get("subgenres", []):
            GraphCreator._process_subgenres(tx, subgenre, child_name)

    @staticmethod
    def _create_genres_tree(tx, data):
        # --- Top Level: 'Music' Root Node ---
        root_name = "Music"
        tx.run("MERGE (:Genre {name: $name})", name=root_name)
        print(f"Created Root Genre: {root_name}")
        
        # --- Second Level: Supreme Genres (Rap, Rock, Metal) ---
        for item in data["music"]:
            supreme_name = item["supreme_genre"]
            
            # 1. Create Supreme Genre and link it to the 'Music' root
            tx.run("MERGE (:Genre {name: $name})", name=supreme_name)
            
            query_supreme_link = """
            MATCH (root:Genre {name: $root_name})
            MATCH (supreme:Genre {name: $supreme_name})
            MERGE (supreme)-[:IS_SUBGENRE]->(root)
            """
            tx.run(query_supreme_link, root_name=root_name, supreme_name=supreme_name)
            print(f"  -> Linked Supreme Genre: {supreme_name}")

            # 2. Start the recursive traversal for all subgenres
            for subgenre_data in item.get("subgenres", []):
                GraphCreator._process_subgenres(tx, subgenre_data, supreme_name)

    @staticmethod
    def _connect_to_earth(tx):
        """Creates the Earth node and connects all existing Region nodes to it."""
        
        print("\nConnecting all Regions to Earth...")
        # 1. Create the single Earth node
        tx.run("MERGE (e:Planet {name: 'Earth'})")

        # 2. Connect all existing Region nodes to the Earth node
        query = """
        MATCH (e:Planet {name: 'Earth'})
        MATCH (r:Region)
        MERGE (r)-[:LOCATED_IN]->(e)
        """
        tx.run(query)
        print("Earth node and PART_OF relationships created successfully.")

    @staticmethod
    def _create_locations_tree(tx, data):
        for region in data["Parts_of_the_World"]:
            region_name = region["name"]
            print(f"Processing Region: {region_name}")
            
            # 1. Create the Region Node
            tx.run("MERGE (:Region {name: $name})", name=region_name)
            
            # Check if 'countries' exists (handling the 'unknown' case)
            if "countries" in region:
                for country in region["countries"]:
                    country_name = country["name"]
                    
                    # 2. Create Country and link to Region
                    query_country = """
                    MATCH (r:Region {name: $region_name})
                    MERGE (c:Country {name: $country_name})
                    MERGE (c)-[:LOCATED_IN]->(r)
                    """
                    tx.run(query_country, region_name=region_name, country_name=country_name)
                    
                    # 3. Create Cities and link to Country
                    for city_name in country["cities"]:
                        query_city = """
                        MATCH (c:Country {name: $country_name})
                        MERGE (city:City {name: $city_name})
                        MERGE (city)-[:LOCATED_IN]->(c)
                        """
                        tx.run(query_city, country_name=country_name, city_name=city_name)
        
        GraphCreator._connect_to_earth(tx)


    def create_genres_tree(self, data):
        """Populates the genre database with the given data."""
        with self.driver.session() as session:
            session.execute_write(self._create_genres_tree, data)

    def create_locations_tree(self, data):
        """Populates the location database with the given data."""
        with self.driver.session() as session:
            session.execute_write(self._create_locations_tree, data)

    def import_track(self, track_data):
        # 1. Parse the features string into a real list
        raw_features = track_data.get('features', '{}')
        parsed_features = self.parse_features_string(raw_features)
        
        # 2. Add this parsed list back to data context so Neo4j can use it
        track_data['parsed_features'] = parsed_features

        with self.driver.session() as session:
            session.execute_write(self._create_nodes, track_data)

            artist_data = track_data.get('artist', {})
            artist_name = artist_data.get('name', track_data.get('artist_name', ''))
            track_id = track_data.get('id')

            # Use pre-computed embeddings from JSON data
            description_chunks = track_data.get('description_chunks', [])
            if description_chunks and artist_name:
                desc_payload = [
                    {
                        "id": f"desc_emb_{artist_name}_{chunk['idx']}",
                        "text": chunk['text'],
                        "vector": chunk['embedding'],
                        "number": chunk['idx']
                    }
                    for chunk in description_chunks
                    if 'idx' in chunk and 'embedding' in chunk
                ]
                if desc_payload:
                    session.execute_write(
                        self._store_embeddings,
                        "Description",
                        f"desc_{artist_name}",
                        desc_payload
                    )

            # Lyrics embeddings from lyrics_chunks
            lyrics_chunks = track_data.get('lyrics_chunks', [])
            if lyrics_chunks and track_id:
                lyrics_payload = [
                    {
                        "id": f"lyrics_emb_{track_id}_{chunk['idx']}",
                        "text": chunk['text'],
                        "vector": chunk['embedding'],
                        "number": chunk['idx']
                    }
                    for chunk in lyrics_chunks
                    if 'idx' in chunk and 'embedding' in chunk
                ]
                if lyrics_payload:
                    session.execute_write(
                        self._store_embeddings,
                        "Lyrics",
                        f"lyrics_{track_id}",
                        lyrics_payload
                    )

    @staticmethod
    def _create_nodes(tx, data):
        artist_data = data.get('artist', {})
        album_data = data.get('album', {})
        
        # ==========================================
        # 1. UPSERT MAIN ARTIST
        # ==========================================
        # Logic: MERGE checks for name.
        # ON CREATE: Sets all data.
        # ON MATCH: Sets data (updates skeleton nodes created by 'features' logic previously)
        query_artist = """
        MERGE (a:Artist {name: $artist_name})
        
        // Always update these properties if we have the main dictionary
        SET 
            a.mbid = $artist_mbid,
            a.url = $artist_url,
            a.founded_year = $founded_year
        
        // 1a. Artist Description (Node & Link)
        WITH a
        MERGE (d:Description {id: 'desc_' + a.name})
        SET d.text = $artist_desc
        MERGE (a)-[:HAS_DESCRIPTION]->(d)

        // 1b. Artist Location (City or Country)
        WITH a
        OPTIONAL MATCH (loc:City {name: $founded_place})
        WITH a, loc
        CALL apoc.do.when(loc IS NOT NULL, 
            'MERGE (a)-[:FOUNDED_IN]->(loc)', 
            'MATCH (c:Country {name: $founded_place}) MERGE (a)-[:FOUNDED_IN]->(c)',
            {a:a, loc:loc, founded_place:$founded_place}) YIELD value
        MATCH (a:Artist {name: "none"}) 
        WITH a
        UNWIND $artist_tags AS tag_name
        MATCH (g:Genre {name: tag_name})
        RETURN g
        """
        fake_tags = ['non_existing_genre']
        try:
            founded_year=int(artist_data.get('founded_year').strip())
        except Exception as e:
            founded_year=-1

        tx.run(query_artist, 
            artist_name=artist_data.get('name', data.get('artist_name')),
            artist_mbid=artist_data.get('mbid'),
            artist_url=artist_data.get('url'),
            founded_year=founded_year,
            artist_desc=artist_data.get('description', ''),
            founded_place=artist_data.get('founded_place', ''),
            artist_tags=fake_tags
        )

        query_genre = """
        MATCH (a:Artist {name: $artistName}) 
        WITH a
        UNWIND $artist_tags AS tag_name
        MATCH (g:Genre {name: tag_name})
        MERGE (a)-[:PERFORMS_GENRE]->(g)
        """

        tags = artist_data.get('tags', [])
        tags = [t.lower() for t in tags] 
        # print("tags:", type(tags),tags)

        tx.run(query_genre,
            artistName=artist_data.get('name', data.get('artist_name')),
            artist_tags=tags)

        # ==========================================
        # 2. HANDLE ALBUM (If present)
        # ==========================================
        if album_data:
            query_album = """
            MATCH (a:Artist {name: $artist_name})
            MERGE (alb:Album {name: $album_name})
            ON CREATE SET 
                alb.mbid = $album_mbid,
                alb.url = $album_url,
                alb.playcount = $album_playcount
            // Also update playcount on match if needed
            ON MATCH SET
                alb.playcount = $album_playcount
            
            MERGE (alb)-[:RELEASED_BY]->(a)
            """
            tx.run(query_album,
                artist_name=artist_data.get('name', data.get('artist_name')),
                album_name=album_data.get('name'),
                album_mbid=album_data.get('mbid'),
                album_url=album_data.get('url'),
                album_playcount=album_data.get('playcount')
            )

        # ==========================================
        # 3. TRACK & FEATURE ARTISTS
        # ==========================================
        query_track = """
        MATCH (a:Artist {name: $artist_name})
        
        MERGE (t:Track {id: $track_id})
        SET 
            t.title = $title,
            t.year = $year,
            t.views = $views
        
        MERGE (t)-[:PERFORMED_BY]->(a)
        
        // Link to Album if exists
        WITH t
        OPTIONAL MATCH (alb:Album {name: $album_name}) WHERE $album_name IS NOT NULL
        FOREACH (_ IN CASE WHEN alb IS NOT NULL THEN [1] ELSE [] END |
            MERGE (t)-[:APPEARS_ON]->(alb)
        )
        
        // Track Genres
        WITH t
        MATCH (g:Genre {name: $tag})
        MERGE (t)-[:HAS_GENRE]->(g)
        
        // Lyrics
        WITH t
        MERGE (l:Lyrics {id: 'lyrics_' + $track_id})
        SET l.text = $lyrics
        MERGE (t)-[:HAS_LYRICS]->(l)

        // 3a. Handle Feature Artists
        // We create them if they don't exist (Skeleton Node), 
        // or match them if they do.
        WITH t
        UNWIND $features AS feature_name
        MERGE (f:Artist {name: feature_name})
        MERGE (t)-[:FEATURING]->(f)
        """

        tx.run(query_track,
            track_id=data.get('id'),
            title=data.get('title'),
            year=data.get('year'),
            views=data.get('views'),
            artist_name=artist_data.get('name', data.get('artist_name')),
            album_name=album_data.get('name') if album_data else None,
            lyrics=data.get('lyrics', ''),
            tag=data.get('tag', 'unknown').lower(),
            features=data.get('parsed_features', [])
        )

    @staticmethod
    def _store_embeddings(tx, target_label: str, target_id: str, embeddings: List[dict]):
        """
        Store embeddings and link them to the target node.
        Expects each embedding dict to have keys: id, vector (list of floats), number.
        """
        if target_label not in {"Lyrics", "Description"}:
            return
        embedding_type = "EmbeddingDescription" if target_label == "Description" else "EmbeddingLyrics"
        query = f"""
        UNWIND $embeddings AS emb
        MERGE (e:{embedding_type} {{id: emb.id}})
        SET e.vector = emb.vector,
            e.text = emb.text,
            e.number = emb.number
        WITH e
        MATCH (t:{target_label} {{id: $target_id}})
        MERGE (t)-[:HAS_EMBEDDING]->(e)
        """
        tx.run(query, target_id=target_id, embeddings=embeddings)



    def create_indexes(self):
        commands = [
            # --- UNIQUE CONSTRAINTS (For Data Integrity) ---
            # Reference Data (Name must be unique)
            "CALL apoc.schema.assert({}, {}, true)",
            "CREATE CONSTRAINT genre_uniq IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
            "CREATE CONSTRAINT region_uniq IF NOT EXISTS FOR (r:Region) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT country_uniq IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT city_uniq IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT planet_uniq IF NOT EXISTS FOR (p:Planet) REQUIRE p.name IS UNIQUE",

            # Entity IDs (Critical for merging and matching)
            "CREATE CONSTRAINT artist_mbid_uniq IF NOT EXISTS FOR (a:Artist) REQUIRE a.mbid IS UNIQUE",
            "CREATE CONSTRAINT album_mbid_uniq IF NOT EXISTS FOR (a:Album) REQUIRE a.mbid IS UNIQUE",
            "CREATE CONSTRAINT track_id_uniq IF NOT EXISTS FOR (t:Track) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT lyrics_id_uniq IF NOT EXISTS FOR (l:Lyrics) REQUIRE l.id IS UNIQUE",
            "CREATE CONSTRAINT desc_id_uniq IF NOT EXISTS FOR (d:Description) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT embed_desc_id_uniq IF NOT EXISTS FOR (e:EmbeddingDescription) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT embed_lyr_id_uniq IF NOT EXISTS FOR (e:EmbeddingLyrics) REQUIRE e.id IS UNIQUE",

            # --- REGULAR INDEXES (For Performance & Sorting) ---
            # Lookups by Name/Title (Non-unique)
            # "CREATE INDEX artist_name_idx IF NOT EXISTS FOR (a:Artist) ON (a.name)",
            # "CREATE INDEX album_name_idx IF NOT EXISTS FOR (a:Album) ON (a.name)",
            # "CREATE INDEX track_title_idx IF NOT EXISTS FOR (t:Track) ON (t.title)",

            "CREATE FULLTEXT INDEX lyrics_fulltext IF NOT EXISTS FOR (l:Lyrics) ON EACH [l.text];",

            "CREATE FULLTEXT INDEX desc_fulltext IF NOT EXISTS FOR (d:Description) ON EACH [d.text];",

            "CREATE FULLTEXT INDEX artist_name_fulltext IF NOT EXISTS FOR (a:Artist) ON EACH [a.name];",
            
            "CREATE FULLTEXT INDEX album_name_fulltext IF NOT EXISTS FOR (a:Album) ON EACH [a.name];",

            "CREATE FULLTEXT INDEX track_title_fulltext IF NOT EXISTS FOR (t:Track) ON EACH [t.title];",

            "CREATE FULLTEXT INDEX genre_fulltext IF NOT EXISTS FOR (g:Genre) ON EACH [g.name];",

            "CREATE FULLTEXT INDEX location_fulltext IF NOT EXISTS FOR (n:Country|City|Region) ON EACH [n.name]",
            
            # Filtering (e.g., "Tracks from 2020")
            "CREATE INDEX artist_founded_idx IF NOT EXISTS FOR (a:Artist) ON (a.founded_year)",
            "CREATE INDEX track_year_idx IF NOT EXISTS FOR (t:Track) ON (t.year)",

            # Sorting (e.g., "Most viewed tracks")
            "CREATE INDEX track_views_idx IF NOT EXISTS FOR (t:Track) ON (t.views)",
            "CREATE INDEX album_playcount_idx IF NOT EXISTS FOR (a:Album) ON (a.playcount)",

            f"""
            CREATE VECTOR INDEX embedding_d_vector IF NOT EXISTS
            FOR (e:EmbeddingDescription) ON (e.vector)
            OPTIONS {{indexConfig: {{
            `vector.dimensions`: {self.embedding_description_dim},
            `vector.similarity_function`: 'cosine'
            }}}}
            """,

            f"""
            CREATE VECTOR INDEX embedding_l_vector IF NOT EXISTS
            FOR (e:EmbeddingLyrics) ON (e.vector)
            OPTIONS {{indexConfig: {{
            `vector.dimensions`: {self.embedding_lyric_dim},
            `vector.similarity_function`: 'cosine'
            }}}}
            """
        ]
        with self.driver.session() as session:
            for cmd in commands:
                print(f"Executing: {cmd.strip().splitlines()[0]}...") # Print first line for brevity
                try:
                    session.run(cmd)
                except Exception as e:
                    print(f"Error executing command: {e}")
            print("\nAll indexes and constraints checked/created.")



def print_time(msg):
    print(f"{msg} in: {b-a:.2f}s")
if __name__ == "__main__":
    # TRACKS_TO_ADD = 100
    start = time.time()
    # dataset_path = "../data_parsing/data/ds2_merged_10000.json"
    dataset_path = "../data_parsing/data/balanced_15k_merged.json"
    dataset_path = "test.json"

    genres_hieararchy_path = "../data/genres_sample.json"
    locations_hieararchy_path = "../data/locations_sample.json"

    uri = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    # uri = "bolt://localhost:7687"
    # auth = ("neo4j", "password")
    
    if not all([uri, username, password]):
        raise ValueError("Neo4j connection environment variables not set.")

    auth = (username, password)
    
    creator = GraphCreator(uri, auth)

    try:
        a = time.time()
        data = creator.read_json(dataset_path)["songs"]
        TRACKS_TO_ADD = len(data)

        b = time.time()
        print_time("Loaded dataset")

        genres_hierarchy = creator.read_json(genres_hieararchy_path)
        countries_hieararchy = creator.read_json(locations_hieararchy_path)

        creator.create_indexes()
        creator.create_genres_tree(genres_hierarchy)
        creator.create_locations_tree(countries_hieararchy)

        print("Starting import...")
        a = time.time()
        batch = 100
        for i, track in enumerate(data):
            creator.import_track(track)
            if i % batch == 0 and i > 0:
                b = time.time()
                print_time(f"Imported {i}/{TRACKS_TO_ADD} tracks")
                a = time.time()
            # print(f"Imported: {track['title']}")
        
    finally:
        creator.close()
        a = start
        b = time.time()
        print_time("Import finished.")