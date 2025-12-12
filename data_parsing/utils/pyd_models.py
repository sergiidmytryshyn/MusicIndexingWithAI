from pydantic import BaseModel
from typing import Optional, List

# Last.fm
class TrackInfo(BaseModel):
    position: Optional[int]
    name: Optional[str]
    duration: Optional[int]
    url: Optional[str]

class AlbumInfo(BaseModel):
    name: str
    mbid: Optional[str]
    url: Optional[str]
    playcount: Optional[int]
    tracks: List[TrackInfo]

class ArtistInfo(BaseModel):
    name: str
    mbid: Optional[str]
    url: Optional[str]
    description: str
    tags: List[str]
    founded_year: Optional[str]
    founded_place: Optional[str]
    albums: List[AlbumInfo]  

# Genius
class GeniusSongInfo(BaseModel):
    title: str
    tag: str
    artist: str
    year: Optional[int]
    views: Optional[int]
    features: Optional[str]
    lyrics: Optional[str]
    id: Optional[int]

class GeniusSongDataset(BaseModel):
    songs: List[GeniusSongInfo]

# Merged model: Genius song with enriched artist and album info
class MergedSongInfo(BaseModel):
    # Original Genius fields
    title: str
    tag: str
    artist_name: str  # Keep original artist name string
    year: Optional[int]
    views: Optional[int]
    features: Optional[str]
    lyrics: Optional[str]
    id: Optional[int]
    # Enriched fields
    artist: Optional[ArtistInfo]  # Full artist info with albums
    album: Optional[AlbumInfo]  # Album info if matched

class MergedSongDataset(BaseModel):
    songs: List[MergedSongInfo]

# Chunk model for lyrics and description chunks
class Chunk(BaseModel):
    idx: int
    text: str
    embedding: List[float]

# Final Song model with chunks
class SongFinal(BaseModel):
    title: str
    tag: str
    artist_name: str  
    year: Optional[int]
    views: Optional[int]
    features: Optional[str]
    lyrics: Optional[str]
    id: Optional[int]
    artist: Optional[ArtistInfo]  
    album: Optional[AlbumInfo]  
    lyrics_chunks: List[Chunk]
    description_chunks: List[Chunk]

class SongFinalDataset(BaseModel):
    songs: List[SongFinal]