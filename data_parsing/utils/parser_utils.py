#!/usr/bin/env python3
"""
Fetch artist info from Last.fm, enrich with MusicBrainz 'founded when/where',
and save to a JSON file.

Data fetched:
- Last.fm: description (bio), tags, top albums and tracklists
- MusicBrainz: founded_year (life-span.begin), founded_place = COUNTRY (area.name)
"""
from __future__ import annotations
import re
import time
from typing import Dict, Any, List, Optional, Tuple
import requests
try:
    from .pyd_models import TrackInfo, ArtistInfo, AlbumInfo
except ImportError:
    from pyd_models import TrackInfo, ArtistInfo, AlbumInfo

# ---------- Config ----------
LASTFM_BASE_URL = "https://ws.audioscrobbler.com/2.0/"
MB_BASE_URL = "https://musicbrainz.org/ws/2"
DEFAULT_ARTIST = "Burzum"
RATE_LIMIT_ERROR_CODE = 29

# ---------- Last.fm helpers ----------
def call_lastfm(method: str, params: Dict[str, Any], api_key: str, max_retries: int = 5, backoff: float = 1.5) -> Dict[str, Any]:
    payload = {"method": method, "api_key": api_key, "format": "json", **params}
    for attempt in range(max_retries):
        try:
            resp = requests.get(LASTFM_BASE_URL, params=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
                continue
            raise RuntimeError(f"Last.fm request failed: {e}")

        if isinstance(data, dict) and "error" in data:
            code = data.get("error")
            msg = data.get("message", "Unknown Last.fm error")
            if code == RATE_LIMIT_ERROR_CODE and attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
                continue
            raise RuntimeError(f"Last.fm API error {code}: {msg}")
        return data
    raise RuntimeError("Exceeded retries calling Last.fm")

def get_artist_info_lastfm(artist: str, api_key: str) -> Dict[str, Any]:
    data = call_lastfm("artist.getInfo", {"artist": artist, "autocorrect": 1}, api_key)
    art = data.get("artist", {}) if isinstance(data, dict) else {}
    bio = art.get("bio") or {}
    description = (bio.get("summary") or bio.get("content") or "").strip()
    tags = [t["name"] for t in (art.get("tags", {}).get("tag") or []) if isinstance(t, dict) and "name" in t]
    return {
        "name": art.get("name", artist),
        "mbid": (art.get("mbid") or None),
        "url": art.get("url"),
        "description": strip_html(description),
        "tags": tags,
    }

def get_top_albums(artist: str, api_key: str, limit: int = 30) -> List[Dict[str, Any]]:
    data = call_lastfm("artist.getTopAlbums", {"artist": artist, "autocorrect": 1, "limit": limit}, api_key)
    albums = (data.get("topalbums", {}) or {}).get("album") or []
    seen: set[str] = set()
    results = []
    for a in albums:
        if not isinstance(a, dict):
            continue
        name = a.get("name")
        mbid = a.get("mbid") or ""
        key = mbid or (name.lower() if name else "")
        if not name or key in seen:
            continue
        seen.add(key)
        results.append({
            "name": name,
            "mbid": mbid or None,
            "url": a.get("url"),
            "playcount": int(a.get("playcount", 0)) if str(a.get("playcount", "0")).isdigit() else None,
        })
    return results

def get_album_tracks(artist: str, album_name: str, api_key: str, mbid: Optional[str] = None) -> List[Dict[str, Any]]:
    params = {"autocorrect": 1}
    if mbid:
        params["mbid"] = mbid
    else:
        params["artist"] = artist
        params["album"] = album_name

    data = call_lastfm("album.getInfo", params, api_key)
    album = data.get("album", {}) if isinstance(data, dict) else {}
    tracks = (album.get("tracks", {}) or {}).get("track") or []
    result = []
    if isinstance(tracks, dict):
        tracks = [tracks]
    for t in tracks:
        if not isinstance(t, dict):
            continue
        result.append({
            "position": int(t.get("@attr", {}).get("rank", 0)) if isinstance(t.get("@attr"), dict) else None,
            "name": t.get("name"),
            "duration": int(t.get("duration", 0)) if str(t.get("duration", "0")).isdigit() else None,
            "url": t.get("url"),
        })
    result.sort(key=lambda x: (x["position"] is None, x["position"] or 0, x["name"] or ""))
    return result

# ---------- MusicBrainz helpers ----------
def mb_get(path: str, params: Dict[str, Any], user_agent: str, max_retries: int = 5, backoff: float = 1.7) -> Dict[str, Any]:
    """
    Robust GET with valid MusicBrainz User-Agent and retry/backoff.
    MusicBrainz requires a UA like: 'MyApp/1.0 ( you@example.com )' or with a project URL.
    """
    headers = {"User-Agent": user_agent}
    url = f"{MB_BASE_URL}/{path}"
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code in (429, 503, 502):
                if attempt < max_retries - 1:
                    time.sleep(backoff ** attempt)
                    continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
                continue
            raise RuntimeError(f"MusicBrainz request failed: {e}")

def get_formation_from_musicbrainz(mbid: Optional[str], artist_name: str, user_agent: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Prefer fetching by MBID; fallback to search by name.
    Returns (founded_year, founded_place_country).

    We intentionally use the 'area' field (country-level) for founded_place.
    Do NOT request 'area'/'begin-area' via inc= — they are core fields.
    """
    if mbid:
        details = mb_get(f"artist/{mbid}", {"fmt": "json"}, user_agent)
    else:
        search = mb_get("artist", {"fmt": "json", "query": f'artist:\"{artist_name}\"'}, user_agent)
        candidates = search.get("artists", []) or []
        if not candidates:
            return None, None
        # Prefer groups, then best score
        candidates.sort(key=lambda a: (
            0 if (a.get("type") or "").lower() == "group" else 1,
            -(a.get("score") or 0)
        ))
        best = candidates[0]
        details = mb_get(f"artist/{best.get('id')}", {"fmt": "json"}, user_agent)

    # life-span.begin -> year
    life = details.get("life-span") or {}
    begin = life.get("begin")
    founded_year = begin[:4] if isinstance(begin, str) and re.match(r"^\d{4}", begin) else None

    # COUNTRY: use area.name (country-level). Ignore begin-area (city-level).
    country = None
    area = details.get("area")
    if isinstance(area, dict) and area.get("name"):
        # If type or ISO codes indicate a country, great; otherwise use name anyway as MB often puts country here.
        if area.get("type") == "Country" or area.get("iso-3166-1-codes"):
            country = area["name"]
        else:
            country = area["name"]

    return founded_year, country

# ---------- Utilities ----------
def strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip()

# ---------- Main ----------
def parse_info(artist: str, api_key: str, album_limit: int, sleep: float, output: str, pretty: bool, mb_user_agent: str):
    if not api_key:
        raise SystemExit("Error: supply a Last.fm API key via --api-key or LASTFM_API_KEY env var.")

    print(f"Fetching Last.fm data for: {artist} ...")
    artist_info = get_artist_info_lastfm(artist, api_key)

    # MusicBrainz formation info (country)
    print("Fetching formation info from MusicBrainz ...")
    try:
        mb_year, mb_country = get_formation_from_musicbrainz(artist_info.get("mbid"), artist_info.get("name") or artist, mb_user_agent)
        artist_info["founded_year"] = mb_year
        artist_info["founded_place"] = mb_country  # country-level only
    except Exception as e:
        artist_info["founded_year"] = None
        artist_info["founded_place"] = None
        print(f"Warning: MusicBrainz lookup failed: {e}")

    # Albums + tracks from Last.fm
    albums = get_top_albums(artist, api_key, limit=album_limit)
    album_objects = []
    for alb in albums:
        try:
            tracks_data = get_album_tracks(artist, alb["name"], api_key, mbid=alb.get("mbid"))
            tracks = [TrackInfo(**track) for track in tracks_data]
            album_objects.append(AlbumInfo(
                name=alb["name"],
                mbid=alb.get("mbid"),
                url=alb.get("url"),
                playcount=alb.get("playcount"),
                tracks=tracks
            ))
        except Exception as e:
            # Skip albums with track fetch errors, or create album with empty tracks
            print(f"Warning: Failed to fetch tracks for album '{alb.get('name')}': {e}")
            album_objects.append(AlbumInfo(
                name=alb["name"],
                mbid=alb.get("mbid"),
                url=alb.get("url"),
                playcount=alb.get("playcount"),
                tracks=[]
            ))
        time.sleep(sleep)
    
    # Return ParsedInfo with proper Pydantic models
    # Create ArtistInfo with albums included
    artist_with_albums = ArtistInfo(**artist_info, albums=album_objects)
    return artist_with_albums
    


