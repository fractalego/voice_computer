#!/usr/bin/env python3
"""
Spotify MCP Server

Provides tools to search for music content on Spotify including:
- Track search
- Artist search  
- Album search
- Playlist search

Uses the Spotify Web API with Client Credentials flow for authentication.
Requires SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from fastmcp import FastMCP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spotify-mcp-server")

# Initialize the MCP server
mcp = FastMCP("Spotify Music Search")

def _get_spotify_client() -> Optional[spotipy.Spotify]:
    """
    Initialize and return a Spotify client using Client Credentials flow.
    
    Returns:
        Spotify client instance or None if credentials are not available
    """
    try:
        client_id = os.getenv('SPOTIFY_CLIENT_ID')
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            logger.error("Spotify credentials not found. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
            return None
        
        # Set up client credentials flow
        client_credentials_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        
        # Create Spotify client
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        
        logger.info("Spotify client initialized successfully")
        return sp
        
    except Exception as e:
        logger.error(f"Failed to initialize Spotify client: {e}")
        return None

def _search_tracks(
    query: str,
    limit: int = 10,
    offset: int = 0,
    market: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for tracks on Spotify.
    
    Args:
        query: Search query string (track name, artist name, etc.)
        limit: Maximum number of results to return (1-50, default: 10)
        offset: Offset for pagination (default: 0)
        market: Market/country code for track availability (ISO 3166-1 alpha-2)
        
    Returns:
        Dictionary containing search results and metadata
    """
    try:
        sp = _get_spotify_client()
        if not sp:
            return {
                "status": "error",
                "error": "Spotify client not available. Check credentials."
            }
        
        # Validate limit
        if limit < 1 or limit > 50:
            limit = min(max(limit, 1), 50)
        
        logger.info(f"Searching for tracks: '{query}' (limit: {limit}, offset: {offset})")
        
        # Perform search
        results = sp.search(
            q=query,
            type='track',
            limit=limit,
            offset=offset,
            market=market
        )
        
        # Process results
        tracks = []
        for item in results['tracks']['items']:
            # Extract artist names
            artists = [artist['name'] for artist in item['artists']]
            
            # Format duration
            duration_ms = item.get('duration_ms', 0)
            duration_minutes = duration_ms // 60000
            duration_seconds = (duration_ms % 60000) // 1000
            duration_formatted = f"{duration_minutes}:{duration_seconds:02d}"
            
            track_info = {
                'id': item['id'],
                'name': item['name'],
                'artists': artists,
                'album': item['album']['name'],
                'duration': duration_formatted,
                'duration_ms': duration_ms,
                'popularity': item.get('popularity', 0),
                'explicit': item.get('explicit', False),
                'preview_url': item.get('preview_url'),
                'external_urls': item.get('external_urls', {}),
                'release_date': item['album'].get('release_date'),
                'available_markets': len(item.get('available_markets', [])),
                'uri': item['uri']
            }
            tracks.append(track_info)
        
        result = {
            "status": "success",
            "query": {
                "search_term": query,
                "limit": limit,
                "offset": offset,
                "market": market
            },
            "total_results": results['tracks']['total'],
            "returned_results": len(tracks),
            "tracks": tracks
        }
        
        logger.info(f"Found {len(tracks)} tracks out of {results['tracks']['total']} total matches")
        return result
        
    except spotipy.exceptions.SpotifyException as e:
        error_msg = f"Spotify API error: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "error_type": "spotify_api"
        }
    except Exception as e:
        error_msg = f"Unexpected error searching tracks: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "error_type": "general"
        }

def _search_artists(
    query: str,
    limit: int = 10,
    offset: int = 0,
    market: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for artists on Spotify.
    
    Args:
        query: Search query string (artist name)
        limit: Maximum number of results to return (1-50, default: 10)
        offset: Offset for pagination (default: 0)
        market: Market/country code for availability (ISO 3166-1 alpha-2)
        
    Returns:
        Dictionary containing search results and metadata
    """
    try:
        sp = _get_spotify_client()
        if not sp:
            return {
                "status": "error",
                "error": "Spotify client not available. Check credentials."
            }
        
        # Validate limit
        if limit < 1 or limit > 50:
            limit = min(max(limit, 1), 50)
        
        logger.info(f"Searching for artists: '{query}' (limit: {limit}, offset: {offset})")
        
        # Perform search
        results = sp.search(
            q=query,
            type='artist',
            limit=limit,
            offset=offset,
            market=market
        )
        
        # Process results
        artists = []
        for item in results['artists']['items']:
            artist_info = {
                'id': item['id'],
                'name': item['name'],
                'followers': item['followers']['total'],
                'popularity': item.get('popularity', 0),
                'genres': item.get('genres', []),
                'external_urls': item.get('external_urls', {}),
                'images': [img for img in item.get('images', [])],
                'uri': item['uri']
            }
            artists.append(artist_info)
        
        result = {
            "status": "success",
            "query": {
                "search_term": query,
                "limit": limit,
                "offset": offset,
                "market": market
            },
            "total_results": results['artists']['total'],
            "returned_results": len(artists),
            "artists": artists
        }
        
        logger.info(f"Found {len(artists)} artists out of {results['artists']['total']} total matches")
        return result
        
    except spotipy.exceptions.SpotifyException as e:
        error_msg = f"Spotify API error: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "error_type": "spotify_api"
        }
    except Exception as e:
        error_msg = f"Unexpected error searching artists: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "error_type": "general"
        }

def _search_albums(
    query: str,
    limit: int = 10,
    offset: int = 0,
    market: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for albums on Spotify.
    
    Args:
        query: Search query string (album name, artist name, etc.)
        limit: Maximum number of results to return (1-50, default: 10)
        offset: Offset for pagination (default: 0)
        market: Market/country code for availability (ISO 3166-1 alpha-2)
        
    Returns:
        Dictionary containing search results and metadata
    """
    try:
        sp = _get_spotify_client()
        if not sp:
            return {
                "status": "error",
                "error": "Spotify client not available. Check credentials."
            }
        
        # Validate limit
        if limit < 1 or limit > 50:
            limit = min(max(limit, 1), 50)
        
        logger.info(f"Searching for albums: '{query}' (limit: {limit}, offset: {offset})")
        
        # Perform search
        results = sp.search(
            q=query,
            type='album',
            limit=limit,
            offset=offset,
            market=market
        )
        
        # Process results
        albums = []
        for item in results['albums']['items']:
            # Extract artist names
            artists = [artist['name'] for artist in item['artists']]
            
            album_info = {
                'id': item['id'],
                'name': item['name'],
                'artists': artists,
                'album_type': item.get('album_type', ''),
                'total_tracks': item.get('total_tracks', 0),
                'release_date': item.get('release_date'),
                'release_date_precision': item.get('release_date_precision'),
                'available_markets': len(item.get('available_markets', [])),
                'external_urls': item.get('external_urls', {}),
                'images': [img for img in item.get('images', [])],
                'uri': item['uri']
            }
            albums.append(album_info)
        
        result = {
            "status": "success",
            "query": {
                "search_term": query,
                "limit": limit,
                "offset": offset,
                "market": market
            },
            "total_results": results['albums']['total'],
            "returned_results": len(albums),
            "albums": albums
        }
        
        logger.info(f"Found {len(albums)} albums out of {results['albums']['total']} total matches")
        return result
        
    except spotipy.exceptions.SpotifyException as e:
        error_msg = f"Spotify API error: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "error_type": "spotify_api"
        }
    except Exception as e:
        error_msg = f"Unexpected error searching albums: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "error_type": "general"
        }

# Create MCP tool wrappers
@mcp.tool()
def search_tracks(
    query: str,
    limit: int = 10,
    offset: int = 0,
    market: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for tracks on Spotify.
    
    This tool searches Spotify's catalog for tracks matching your query.
    You can search by track name, artist name, album name, or any combination.
    
    Args:
        query: Search query (e.g., "Bohemian Rhapsody Queen", "Taylor Swift", "Abbey Road")
        limit: Maximum number of results to return (1-50, default: 10)
        offset: Offset for pagination to get more results (default: 0)
        market: Country code for market-specific results (e.g., "US", "GB", "DE")
        
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - query: Details about the search parameters
        - total_results: Total number of tracks matching the search
        - returned_results: Number of tracks returned in this response
        - tracks: List of track information including name, artists, album, duration, etc.
        
    Examples:
        search_tracks("Bohemian Rhapsody") - Find the famous Queen song
        search_tracks("Taylor Swift folklore", limit=5) - Get 5 tracks from folklore album
        search_tracks("jazz piano", market="US") - Find jazz piano tracks available in US
    """
    return _search_tracks(query, limit, offset, market)

@mcp.tool()
def search_artists(
    query: str,
    limit: int = 10,
    offset: int = 0,
    market: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for artists on Spotify.
    
    This tool searches Spotify's catalog for artists matching your query.
    
    Args:
        query: Artist name or search query (e.g., "Taylor Swift", "The Beatles")
        limit: Maximum number of results to return (1-50, default: 10)
        offset: Offset for pagination to get more results (default: 0)
        market: Country code for market-specific results (e.g., "US", "GB", "DE")
        
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - query: Details about the search parameters
        - total_results: Total number of artists matching the search
        - returned_results: Number of artists returned in this response
        - artists: List of artist information including name, followers, genres, popularity, etc.
        
    Examples:
        search_artists("Taylor Swift") - Find Taylor Swift
        search_artists("indie rock bands", limit=15) - Get 15 indie rock artists
    """
    return _search_artists(query, limit, offset, market)

@mcp.tool()
def search_albums(
    query: str,
    limit: int = 10,
    offset: int = 0,
    market: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for albums on Spotify.
    
    This tool searches Spotify's catalog for albums matching your query.
    You can search by album name, artist name, or any combination.
    
    Args:
        query: Search query (e.g., "Abbey Road", "Taylor Swift folklore", "Pink Floyd")
        limit: Maximum number of results to return (1-50, default: 10)
        offset: Offset for pagination to get more results (default: 0)
        market: Country code for market-specific results (e.g., "US", "GB", "DE")
        
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - query: Details about the search parameters
        - total_results: Total number of albums matching the search
        - returned_results: Number of albums returned in this response
        - albums: List of album information including name, artists, release date, track count, etc.
        
    Examples:
        search_albums("Dark Side of the Moon") - Find Pink Floyd's famous album
        search_albums("Taylor Swift", limit=20) - Get 20 Taylor Swift albums
    """
    return _search_albums(query, limit, offset, market)

@mcp.tool()
def search_all_music(
    query: str,
    limit_each: int = 5
) -> Dict[str, Any]:
    """
    Search for tracks, artists, and albums simultaneously on Spotify.
    
    This tool performs a comprehensive search across all music categories
    and returns results for tracks, artists, and albums in a single response.
    
    Args:
        query: Search query (e.g., "Queen", "jazz", "80s rock")
        limit_each: Maximum number of results per category (1-20, default: 5)
        
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - query: Details about the search parameters
        - tracks: Track search results
        - artists: Artist search results
        - albums: Album search results
        
    Examples:
        search_all_music("Queen") - Get tracks, artists, and albums related to Queen
        search_all_music("jazz piano", limit_each=3) - Get 3 results per category
    """
    try:
        # Validate limit
        if limit_each < 1 or limit_each > 20:
            limit_each = min(max(limit_each, 1), 20)
        
        # Perform all searches
        tracks_result = _search_tracks(query, limit=limit_each)
        artists_result = _search_artists(query, limit=limit_each)
        albums_result = _search_albums(query, limit=limit_each)
        
        # Check if all searches were successful
        all_successful = (
            tracks_result.get("status") == "success" and
            artists_result.get("status") == "success" and
            albums_result.get("status") == "success"
        )
        
        result = {
            "status": "success" if all_successful else "partial_success",
            "query": {
                "search_term": query,
                "limit_each": limit_each
            },
            "tracks": tracks_result,
            "artists": artists_result,
            "albums": albums_result
        }
        
        if not all_successful:
            result["warnings"] = "Some searches may have failed. Check individual results."
        
        logger.info(f"Comprehensive search for '{query}' completed")
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error in comprehensive search: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "error_type": "general"
        }

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()