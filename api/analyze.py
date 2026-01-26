"""Analyze YouTube video for peaks - Vercel Serverless Function"""

import json
import re
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler
import urllib.request


def extract_video_id(url: str) -> str | None:
    """Extract video ID from YouTube URL"""
    patterns = [
        r"(?:v=|/v/|youtu\.be/|/embed/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_heatmap(video_id: str) -> list:
    """Fetch heatmap data from YouTube"""
    api_url = "https://www.youtube.com/youtubei/v1/next"

    payload = {
        "context": {
            "client": {
                "clientName": "WEB",
                "clientVersion": "2.20240101.00.00"
            }
        },
        "videoId": video_id
    }

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode())

    markers = []

    # Method 1: frameworkUpdates -> entityBatchUpdate -> mutations
    try:
        mutations = data.get("frameworkUpdates", {}).get("entityBatchUpdate", {}).get("mutations", [])
        for mutation in mutations:
            markers_entity = mutation.get("payload", {}).get("macroMarkersListEntity")
            if markers_entity:
                markers_list = markers_entity.get("markersList", {})
                for marker in markers_list.get("markers", []):
                    markers.append({
                        "start_ms": int(marker.get("startMillis", 0)),
                        "duration_ms": int(marker.get("durationMillis", 0)),
                        "intensity": float(marker.get("intensityScoreNormalized", 0))
                    })
                if markers:
                    return markers
    except Exception:
        pass

    # Method 2: playerOverlays path (fallback)
    try:
        decorations = data.get("playerOverlays", {}).get("playerOverlayRenderer", {}).get("decoratedPlayerBarRenderer", {}).get("decoratedPlayerBarRenderer", {}).get("playerBar", {}).get("multiMarkersPlayerBarRenderer", {}).get("markersMap", [])
        for marker_map in decorations:
            if marker_map.get("key") == "HEATSEEKER":
                heatmarkers = marker_map.get("value", {}).get("heatmap", {}).get("heatmapRenderer", {}).get("heatMarkers", [])
                for hm in heatmarkers:
                    renderer = hm.get("heatMarkerRenderer", {})
                    markers.append({
                        "start_ms": int(renderer.get("timeRangeStartMillis", 0)),
                        "duration_ms": int(renderer.get("markerDurationMillis", 0)),
                        "intensity": float(renderer.get("heatMarkerIntensityScoreNormalized", 0))
                    })
    except Exception:
        pass

    return markers


def detect_peaks(markers: list, threshold: float = 0.5) -> list:
    """Detect peak moments from heatmap data"""
    if not markers:
        return []

    # Filter above threshold
    hot_segments = [m for m in markers if m["intensity"] >= threshold]
    if not hot_segments:
        return []

    # Group consecutive segments
    peaks = []
    current_group = [hot_segments[0]]

    for seg in hot_segments[1:]:
        prev = current_group[-1]
        if seg["start_ms"] <= prev["start_ms"] + prev["duration_ms"] + 1000:
            current_group.append(seg)
        else:
            peaks.append(current_group)
            current_group = [seg]
    peaks.append(current_group)

    # Calculate peak info
    results = []
    for group in peaks:
        start_ms = group[0]["start_ms"]
        end_ms = group[-1]["start_ms"] + group[-1]["duration_ms"]
        avg_intensity = sum(s["intensity"] for s in group) / len(group)

        results.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "intensity": avg_intensity
        })

    # Sort by intensity
    results.sort(key=lambda x: x["intensity"], reverse=True)
    return results[:10]


def format_time(ms: int) -> str:
    """Format milliseconds as MM:SS"""
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight request"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            query = parse_qs(urlparse(self.path).query)
            url = query.get('url', [''])[0]

            if not url:
                self.wfile.write(json.dumps({'error': 'URLが指定されていません'}).encode())
                return

            video_id = extract_video_id(url)
            if not video_id:
                self.wfile.write(json.dumps({'error': '有効なYouTube URLではありません'}).encode())
                return

            markers = fetch_heatmap(video_id)
            if not markers:
                self.wfile.write(json.dumps({'error': 'Heatmapデータが見つかりませんでした'}).encode())
                return

            peaks = detect_peaks(markers)
            if not peaks:
                self.wfile.write(json.dumps({'error': '盛り上がりポイントが見つかりませんでした'}).encode())
                return

            result = []
            for p in peaks:
                result.append({
                    'time_range': f"{format_time(p['start_ms'])} - {format_time(p['end_ms'])}",
                    'score': p['intensity'],
                    'start': p['start_ms'] / 1000,
                    'end': p['end_ms'] / 1000
                })

            self.wfile.write(json.dumps({'peaks': result}).encode())

        except Exception as e:
            self.wfile.write(json.dumps({'error': str(e)}).encode())
