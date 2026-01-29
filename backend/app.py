"""YouTube Clip Generator Backend API with Hook-First Support"""

import os
import json
import subprocess
import tempfile
import uuid
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import urllib.request
from typing import Optional

import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

# Groq client（遅延読み込み）
_groq_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            _groq_client = Groq(api_key=groq_api_key)
    return _groq_client


class GenerateRequest(BaseModel):
    url: str
    start: float
    end: float
    add_subtitle: bool = False


class HookFirstRequest(BaseModel):
    url: str
    peak_start: float
    peak_end: float
    hook_duration: float = 5.0
    context_duration: float = 40.0
    max_total_duration: float = 60.0
    use_llm: bool = True


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


def transcribe_audio(audio_path: str) -> list:
    """Transcribe audio using Groq's Whisper API"""
    client = get_groq_client()
    if not client:
        # Groq APIキーがない場合は空のセグメントを返す
        return []

    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
                language="ja"
            )

        segments = []
        # Handle both object and dict response formats
        raw_segments = None
        if hasattr(transcription, 'segments'):
            raw_segments = transcription.segments
        elif isinstance(transcription, dict) and 'segments' in transcription:
            raw_segments = transcription['segments']

        if raw_segments:
            for seg in raw_segments:
                if isinstance(seg, dict):
                    segments.append({
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "text": seg.get("text", "").strip()
                    })
                else:
                    segments.append({
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip() if seg.text else ""
                    })
        return segments
    except Exception as e:
        print(f"Transcription error: {e}")
        return []


def find_sentence_boundary(segments: list, target_time: float, direction: str = "before") -> float:
    """Find the nearest sentence boundary"""
    sentence_endings = ["。", "！", "？", "!", "?", ".", "…"]

    if direction == "before":
        # 指定時間より前で、文末で終わるセグメントを探す
        for seg in reversed(segments):
            if seg["end"] <= target_time:
                if any(seg["text"].endswith(e) for e in sentence_endings):
                    return seg["end"]
        # 見つからない場合は最も近いセグメント終了時間
        for seg in reversed(segments):
            if seg["end"] <= target_time:
                return seg["end"]
    else:
        # 指定時間より後で、文末で終わるセグメントを探す
        for seg in segments:
            if seg["start"] >= target_time:
                if any(seg["text"].endswith(e) for e in sentence_endings):
                    return seg["end"]
        # 見つからない場合は最も近いセグメント終了時間
        for seg in segments:
            if seg["start"] >= target_time:
                return seg["end"]

    return target_time


def get_text_in_range(segments: list, start: float, end: float) -> str:
    """Get transcript text within a time range"""
    texts = []
    for seg in segments:
        if seg["start"] >= start and seg["end"] <= end:
            texts.append(seg["text"])
        elif seg["start"] < end and seg["end"] > start:
            texts.append(seg["text"])
    return " ".join(texts)


def analyze_hook_with_llm(hook_text: str, context_text: str) -> dict:
    """Use LLM to analyze if hook is good"""
    client = get_groq_client()

    if not client:
        # LLM APIキーがない場合はデフォルトの判定
        return {
            "is_good_hook": True,
            "reason": "LLM分析スキップ（APIキー未設定）",
            "suggested_adjustment": None
        }

    try:

        prompt = f"""以下のフック（冒頭）テキストを評価してください。

【フック】
{hook_text}

【その後の文脈】
{context_text[:500]}

評価基準:
1. フックだけを見て「何が起きたか」が伝わるか
2. 文が途中で切れていないか
3. 視聴者の興味を引くか

JSON形式で回答:
{{
    "is_good_hook": true/false,
    "reason": "理由",
    "suggested_adjustment": "改善案（あれば）"
}}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        result_text = response.choices[0].message.content
        # JSONを抽出
        json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        return {"is_good_hook": True, "reason": "LLM応答解析失敗", "suggested_adjustment": None}

    except Exception as e:
        return {"is_good_hook": True, "reason": f"LLM分析エラー: {str(e)}", "suggested_adjustment": None}


def extract_audio(video_path: str, output_path: str):
    """Extract audio from video"""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def cut_video(input_path: str, output_path: str, start: float, duration: float):
    """Cut video segment"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def concatenate_videos(video_paths: list, output_path: str):
    """Concatenate multiple videos"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for path in video_paths:
            f.write(f"file '{path}'\n")
        list_file = f.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    finally:
        os.unlink(list_file)


@app.get("/")
def root():
    return {"status": "ok", "service": "YouTube Clip Generator API with Hook-First"}


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
    """Generate a simple clip from YouTube video"""
    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "processing", "progress": 0}

    try:
        jobs[job_id]["step"] = "downloading"

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")

            # yt-dlp download
            ydl_opts = {
                'format': 'best[height<=720]/best',
                'outtmpl': video_path,
                'quiet': True,
                'no_warnings': True,
                'extractor_args': {'youtube': {'player_client': ['ios', 'web']}},
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
                },
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            except Exception as e:
                raise Exception(f"Download failed: {str(e)}")

            jobs[job_id]["step"] = "cutting"

            # Cut video
            output_filename = f"clip_{video_id}_{job_id}.mp4"
            output_path = OUTPUT_DIR / output_filename

            duration = request.end - request.start
            cut_video(video_path, str(output_path), request.start, duration)

            jobs[job_id]["status"] = "completed"
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


@app.post("/generate-hook-first")
def generate_hook_first(request: HookFirstRequest):
    """Generate a hook-first style clip

    Structure:
    1. Hook (5秒) - 盛り上がりの最高潮
    2. Context (30-40秒) - なぜそうなったか
    3. Climax (残り) - 再びフック部分へ
    """
    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "processing", "progress": 0, "step": "initializing"}

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            audio_path = os.path.join(tmpdir, "audio.wav")

            # 1. Download video
            jobs[job_id]["step"] = "downloading"
            jobs[job_id]["progress"] = 10

            ydl_opts = {
                'format': 'best[ext=mp4][height<=720]/best[ext=mp4]/best',
                'outtmpl': video_path,
                'quiet': True,
                'no_warnings': True,
                'extractor_args': {'youtube': {'player_client': ['ios', 'web']}},
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
                },
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

            # 2. Extract audio and transcribe
            jobs[job_id]["step"] = "transcribing"
            jobs[job_id]["progress"] = 30

            extract_audio(video_path, audio_path)
            segments = transcribe_audio(audio_path)

            # 3. Calculate hook-first structure
            jobs[job_id]["step"] = "analyzing"
            jobs[job_id]["progress"] = 50

            peak_start = request.peak_start
            peak_end = request.peak_end
            hook_duration = request.hook_duration
            context_duration = request.context_duration

            # フックの開始位置（ピークの少し前から）
            hook_start = max(0, peak_end - hook_duration)

            # 文の区切りに合わせる
            hook_start = find_sentence_boundary(segments, hook_start, "before")
            hook_end = find_sentence_boundary(segments, peak_end, "after")

            # 実際のフック長を計算
            actual_hook_duration = hook_end - hook_start
            if actual_hook_duration < 2:
                hook_end = hook_start + hook_duration

            # コンテキストの開始位置（フックの前）
            context_start = max(0, hook_start - context_duration)
            context_start = find_sentence_boundary(segments, context_start, "before")
            context_end = hook_start

            # LLMでフック品質を分析
            if request.use_llm:
                jobs[job_id]["step"] = "llm_analyzing"
                hook_text = get_text_in_range(segments, hook_start, hook_end)
                context_text = get_text_in_range(segments, context_start, context_end)
                llm_result = analyze_hook_with_llm(hook_text, context_text)
                jobs[job_id]["llm_analysis"] = llm_result

            # 4. Cut and concatenate videos
            jobs[job_id]["step"] = "cutting"
            jobs[job_id]["progress"] = 70

            hook_path = os.path.join(tmpdir, "hook.mp4")
            context_path = os.path.join(tmpdir, "context.mp4")
            climax_path = os.path.join(tmpdir, "climax.mp4")

            # Hook部分を切り出し
            cut_video(video_path, hook_path, hook_start, hook_end - hook_start)

            # Context部分を切り出し
            cut_video(video_path, context_path, context_start, context_end - context_start)

            # Climax部分（フックの再生）を切り出し
            cut_video(video_path, climax_path, hook_start, hook_end - hook_start)

            # 5. Concatenate: Hook -> Context -> Climax
            jobs[job_id]["step"] = "concatenating"
            jobs[job_id]["progress"] = 90

            output_filename = f"hookfirst_{video_id}_{job_id}.mp4"
            output_path = OUTPUT_DIR / output_filename

            concatenate_videos([hook_path, context_path, climax_path], str(output_path))

            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            file_size = output_path.stat().st_size

            return {
                "job_id": job_id,
                "status": "completed",
                "filename": output_filename,
                "file_size": file_size,
                "download_url": f"/download/{output_filename}",
                "structure": {
                    "hook": {"start": hook_start, "end": hook_end, "duration": hook_end - hook_start},
                    "context": {"start": context_start, "end": context_end, "duration": context_end - context_start},
                    "climax": {"start": hook_start, "end": hook_end, "duration": hook_end - hook_start}
                },
                "total_duration": (hook_end - hook_start) * 2 + (context_end - context_start),
                "llm_analysis": jobs[job_id].get("llm_analysis")
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


# Cleanup old files
@app.on_event("startup")
async def cleanup_old_files():
    import time
    for file in OUTPUT_DIR.glob("*.mp4"):
        if time.time() - file.stat().st_mtime > 3600:
            file.unlink()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
