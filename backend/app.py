"""YouTube Clip Generator Backend API"""

import os
import json
import subprocess
import tempfile
import uuid
import base64
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import urllib.request

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="YouTube Clip Generator API")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 一時ファイル保存用ディレクトリ
OUTPUT_DIR = Path("/tmp/clips")
OUTPUT_DIR.mkdir(exist_ok=True)

# ジョブ状態管理
jobs = {}


class GenerateRequest(BaseModel):
    url: str
    start: float
    end: float
    add_subtitle: bool = False


class AnalyzeRequest(BaseModel):
    url: str


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
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
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

    # Method 1: frameworkUpdates
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

    # Method 2: playerOverlays (fallback)
    try:
        decorations = data.get("playerOverlays", {}).get("playerOverlayRenderer", {}).get(
            "decoratedPlayerBarRenderer", {}).get("decoratedPlayerBarRenderer", {}).get(
            "playerBar", {}).get("multiMarkersPlayerBarRenderer", {}).get("markersMap", [])
        for marker_map in decorations:
            if marker_map.get("key") == "HEATSEEKER":
                heatmarkers = marker_map.get("value", {}).get("heatmap", {}).get(
                    "heatmapRenderer", {}).get("heatMarkers", [])
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

    hot_segments = [m for m in markers if m["intensity"] >= threshold]
    if not hot_segments:
        return []

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

    results.sort(key=lambda x: x["intensity"], reverse=True)
    return results[:10]


def format_time(ms: int) -> str:
    """Format milliseconds as MM:SS"""
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


@app.get("/")
def root():
    return {"status": "ok", "service": "YouTube Clip Generator API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Analyze YouTube video for peaks"""
    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    markers = fetch_heatmap(video_id)
    if not markers:
        raise HTTPException(status_code=404, detail="No heatmap data found")

    peaks = detect_peaks(markers)
    if not peaks:
        raise HTTPException(status_code=404, detail="No peaks found")

    result = []
    for p in peaks:
        result.append({
            "time_range": f"{format_time(p['start_ms'])} - {format_time(p['end_ms'])}",
            "score": p["intensity"],
            "start": p["start_ms"] / 1000,
            "end": p["end_ms"] / 1000
        })

    return {"peaks": result}


@app.post("/generate")
def generate_clip(request: GenerateRequest):
    """Generate a clip from YouTube video"""
    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "processing", "progress": 0}

    try:
        # 1. Download video
        jobs[job_id]["progress"] = 10
        jobs[job_id]["step"] = "downloading"

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")

            # yt-dlp download
            cmd = [
                "yt-dlp",
                "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
                "--merge-output-format", "mp4",
                "-o", video_path,
                f"https://www.youtube.com/watch?v={video_id}"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise Exception(f"Download failed: {result.stderr}")

            jobs[job_id]["progress"] = 50
            jobs[job_id]["step"] = "cutting"

            # 2. Cut video
            output_filename = f"clip_{video_id}_{job_id}.mp4"
            output_path = OUTPUT_DIR / output_filename

            duration = request.end - request.start
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(request.start),
                "-i", video_path,
                "-t", str(duration),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "fast",
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise Exception(f"FFmpeg failed: {result.stderr}")

            jobs[job_id]["progress"] = 100
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["filename"] = output_filename

            # Get file size
            file_size = output_path.stat().st_size

            return {
                "job_id": job_id,
                "status": "completed",
                "filename": output_filename,
                "file_size": file_size,
                "download_url": f"/download/{output_filename}"
            }

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job/{job_id}")
def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/download/{filename}")
def download_clip(filename: str):
    """Download generated clip"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename
    )


# Cleanup old files (runs periodically)
@app.on_event("startup")
async def cleanup_old_files():
    import time
    for file in OUTPUT_DIR.glob("*.mp4"):
        if time.time() - file.stat().st_mtime > 3600:  # 1 hour
            file.unlink()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
