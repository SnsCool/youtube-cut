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

# PO Token Server URL (for YouTube bot detection bypass)
POT_SERVER_URL = os.environ.get("POT_SERVER_URL", "")

# Set bgutil plugin environment variable at module level
if POT_SERVER_URL:
    os.environ['BGUTIL_POT_HTTP_BASE_URL'] = POT_SERVER_URL
    print(f"PO Token server configured: {POT_SERVER_URL}")


def get_ydl_opts(video_path: str) -> dict:
    """Get yt-dlp options with PO Token support"""
    opts = {
        'outtmpl': video_path.replace('.mp4', '.%(ext)s'),
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        },
        'socket_timeout': 30,
        'retries': 5,
        'extractor_retries': 3,
    }

    # Format selection with fallback to format 18 (360p mp4, no PO Token required)
    # Priority: 720p > 480p > 360p > format 18 (guaranteed to work without PO Token)
    opts['format'] = 'bestvideo[height<=720]+bestaudio/best[height<=720]/best/18'

    # PO Token configuration for web client (bypasses bot detection)
    # Using bgutil-ytdlp-pot-provider plugin with HTTP server
    if POT_SERVER_URL:
        # Use web client with PO Token (best compatibility with PO Token)
        # Fallback to ios/android if web fails
        opts['extractor_args'] = {
            'youtube': {
                'player_client': ['web', 'ios', 'android'],
            },
            # New extractor args format for bgutil-ytdlp-pot-provider v1.0.0+
            'youtubepot-bgutilhttp': {
                'base_url': POT_SERVER_URL,
            }
        }
        print(f"Using PO Token with web client: {POT_SERVER_URL}")
    else:
        # No PO Token server - use default client (uses format 18 as fallback)
        # Format 18 is a progressive format that doesn't require PO Token
        opts['extractor_args'] = {
            'youtube': {
                'player_client': ['default'],
            }
        }
        print("No PO Token server - using default client with format 18 fallback")

    return opts


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


def find_sentence_boundary(segments: list, target_time: float, direction: str = "before", max_distance: float = 10.0) -> float:
    """Find the nearest sentence boundary within max_distance seconds

    Args:
        segments: List of transcript segments
        target_time: Target time to find boundary near
        direction: "before" or "after"
        max_distance: Maximum distance in seconds to search (default 10 seconds)
    """
    sentence_endings = ["。", "！", "？", "!", "?", ".", "…"]

    if direction == "before":
        min_time = target_time - max_distance
        # 指定時間より前で、文末で終わるセグメントを探す（max_distance以内）
        for seg in reversed(segments):
            if seg["end"] <= target_time and seg["end"] >= min_time:
                if any(seg["text"].endswith(e) for e in sentence_endings):
                    return seg["end"]
        # 見つからない場合は最も近いセグメント終了時間（max_distance以内）
        for seg in reversed(segments):
            if seg["end"] <= target_time and seg["end"] >= min_time:
                return seg["end"]
    else:
        max_time = target_time + max_distance
        # 指定時間より後で、文末で終わるセグメントを探す（max_distance以内）
        for seg in segments:
            if seg["start"] >= target_time and seg["end"] <= max_time:
                if any(seg["text"].endswith(e) for e in sentence_endings):
                    return seg["end"]
        # 見つからない場合は最も近いセグメント終了時間（max_distance以内）
        for seg in segments:
            if seg["start"] >= target_time and seg["end"] <= max_time:
                return seg["end"]

    # max_distance以内に見つからない場合はtarget_timeをそのまま返す
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


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_audio_chunk(video_path: str, output_path: str, start: float = 0, duration: float = 600):
    """Extract audio chunk from video

    Args:
        video_path: Input video path
        output_path: Output audio path (MP3)
        start: Start time in seconds
        duration: Duration in seconds (default 600 = 10 minutes)
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "32k",  # Low bitrate for smaller file size (~2MB per 10min)
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def transcribe_audio_chunked(video_path: str, tmpdir: str, chunk_duration: int = 600) -> list:
    """Transcribe full video by processing in chunks

    Args:
        video_path: Input video path
        tmpdir: Temporary directory for chunk files
        chunk_duration: Duration of each chunk in seconds (default 600 = 10 minutes)

    Returns:
        List of transcript segments with adjusted timestamps
    """
    client = get_groq_client()
    if not client:
        return []

    # Get total video duration
    total_duration = get_video_duration(video_path)
    print(f"Video duration: {total_duration:.1f} seconds")

    all_segments = []
    chunk_index = 0

    # Process video in chunks
    for start_time in range(0, int(total_duration), chunk_duration):
        chunk_path = os.path.join(tmpdir, f"chunk_{chunk_index}.mp3")
        actual_duration = min(chunk_duration, total_duration - start_time)

        print(f"Processing chunk {chunk_index}: {start_time}s - {start_time + actual_duration}s")

        # Extract audio chunk
        extract_audio_chunk(video_path, chunk_path, start=start_time, duration=actual_duration)

        # Transcribe chunk
        try:
            with open(chunk_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(f"chunk_{chunk_index}.mp3", audio_file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    language="ja"
                )

            # Extract segments and adjust timestamps
            raw_segments = None
            if hasattr(transcription, 'segments'):
                raw_segments = transcription.segments
            elif isinstance(transcription, dict) and 'segments' in transcription:
                raw_segments = transcription['segments']

            if raw_segments:
                for seg in raw_segments:
                    if isinstance(seg, dict):
                        all_segments.append({
                            "start": seg.get("start", 0) + start_time,  # Adjust timestamp
                            "end": seg.get("end", 0) + start_time,
                            "text": seg.get("text", "").strip()
                        })
                    else:
                        all_segments.append({
                            "start": seg.start + start_time,  # Adjust timestamp
                            "end": seg.end + start_time,
                            "text": seg.text.strip() if seg.text else ""
                        })

        except Exception as e:
            print(f"Chunk {chunk_index} transcription error: {e}")

        # Clean up chunk file
        if os.path.exists(chunk_path):
            os.remove(chunk_path)

        chunk_index += 1

    print(f"Total segments transcribed: {len(all_segments)}")
    return all_segments


def extract_audio(video_path: str, output_path: str):
    """Extract full audio from video (for backward compatibility)"""
    extract_audio_chunk(video_path, output_path, start=0, duration=99999)


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
    return {"status": "ok", "service": "YouTube Clip Generator API with Hook-First", "version": "2.1.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/debug/pot")
def debug_pot():
    """Debug PO Token server connectivity and yt-dlp configuration"""
    import urllib.request
    import urllib.error

    result = {
        "pot_server_url": POT_SERVER_URL,
        "bgutil_env": os.environ.get("BGUTIL_POT_HTTP_BASE_URL", "not set"),
        "pot_server_reachable": False,
        "pot_server_response": None,
        "yt_dlp_version": yt_dlp.version.__version__,
        "pot_plugin_loaded": False,
        "pot_token_test": None,
        "error": None
    }

    # Check if bgutil plugin is loaded
    try:
        from yt_dlp.extractor.youtube import YoutubeIE
        # Check for PO Token provider in extractor
        result["pot_plugin_loaded"] = hasattr(YoutubeIE, '_pot_providers') or 'bgutil' in str(dir(YoutubeIE))
    except Exception as e:
        result["pot_plugin_error"] = str(e)

    if POT_SERVER_URL:
        # Test GET on root
        try:
            req = urllib.request.Request(f"{POT_SERVER_URL}/", method="GET")
            with urllib.request.urlopen(req, timeout=10) as response:
                result["pot_server_reachable"] = True
                result["pot_server_response"] = response.status
        except urllib.error.HTTPError as e:
            result["pot_server_reachable"] = True
            result["pot_server_response"] = e.code
        except Exception as e:
            result["error"] = str(e)

        # Test POST to /get_pot endpoint (common endpoint pattern)
        try:
            test_data = json.dumps({"video_id": "test", "data_sync_id": "test"}).encode()
            req = urllib.request.Request(
                f"{POT_SERVER_URL}/get_pot",
                data=test_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result["pot_token_test"] = "success"
                result["pot_token_response"] = response.read().decode()[:200]
        except urllib.error.HTTPError as e:
            result["pot_token_test"] = f"HTTP {e.code}"
            try:
                result["pot_token_response"] = e.read().decode()[:200]
            except:
                pass
        except Exception as e:
            result["pot_token_test"] = f"error: {str(e)}"

    return result


@app.get("/debug/test-download/{video_id}")
def debug_test_download(video_id: str):
    """Test video download with verbose output"""
    import io
    import sys

    result = {
        "video_id": video_id,
        "success": False,
        "error": None,
        "yt_dlp_output": None
    }

    # Capture yt-dlp output
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test.mp4")

        # Get opts with verbose mode
        opts = get_ydl_opts(video_path)
        opts['quiet'] = False
        opts['verbose'] = True
        opts['no_warnings'] = False

        # Capture output
        output_buffer = io.StringIO()

        class MyLogger:
            def debug(self, msg):
                output_buffer.write(f"[debug] {msg}\n")
            def info(self, msg):
                output_buffer.write(f"[info] {msg}\n")
            def warning(self, msg):
                output_buffer.write(f"[warning] {msg}\n")
            def error(self, msg):
                output_buffer.write(f"[error] {msg}\n")

        opts['logger'] = MyLogger()

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                # Just extract info, don't download
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                result["success"] = True
                result["title"] = info.get("title", "Unknown")
                result["formats_count"] = len(info.get("formats", []))
        except Exception as e:
            result["error"] = str(e)

        result["yt_dlp_output"] = output_buffer.getvalue()[-2000:]  # Last 2000 chars

    return result


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


def analyze_peaks_with_llm(segments: list, video_duration: float = None) -> list:
    """Use LLM to analyze transcript and find peak moments"""
    client = get_groq_client()

    if not client or not segments:
        return []

    # セグメントをテキストに変換（時間情報付き）
    transcript_with_time = []
    for seg in segments:
        minutes = int(seg["start"] // 60)
        seconds = int(seg["start"] % 60)
        transcript_with_time.append(f"[{minutes:02d}:{seconds:02d}] {seg['text']}")

    full_transcript = "\n".join(transcript_with_time)

    # トランスクリプトが長すぎる場合は分割
    if len(full_transcript) > 8000:
        full_transcript = full_transcript[:8000] + "\n...(省略)"

    prompt = f"""以下はYouTube動画（令和の虎など討論・プレゼン番組）の文字起こしです。
ショート動画の「フック」として使える最高潮の瞬間を3つ以内で特定してください。

【文字起こし】
{full_transcript}

【良いフックの条件】
1. 結論・決定の瞬間（「ALL達成」「NOTHING」「○○万円出します」など）
2. 感情が爆発する瞬間（怒り、驚き、感動の声）
3. 印象的な一言（名言、衝撃発言）
4. 緊張が最高潮に達する瞬間

【悪いフックの例】
- 説明や前置きの部分
- 淡々とした会話
- 文脈がないと意味が分からない発言

JSON形式で回答:
{{
    "peaks": [
        {{
            "start_time": "MM:SS形式（フックの開始）",
            "end_time": "MM:SS形式（フックの終了、開始から5-8秒後）",
            "reason": "このシーンが最高潮である理由（10文字以内）",
            "score": 0.0-1.0
        }}
    ]
}}

重要:
- start_timeからend_timeは必ず5〜8秒以内にしてください（厳守）
- 最も盛り上がる「瞬間」のみを特定（前後の文脈は含めない）
- フックは短く印象的に。長すぎるとNGです"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )

        result_text = response.choices[0].message.content

        # JSONを抽出
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            data = json.loads(json_match.group())
            peaks = data.get("peaks", [])

            # 時間をパースして秒に変換
            parsed_peaks = []
            for p in peaks:
                try:
                    start_parts = p["start_time"].split(":")
                    end_parts = p["end_time"].split(":")

                    start_sec = int(start_parts[0]) * 60 + int(start_parts[1])
                    end_sec = int(end_parts[0]) * 60 + int(end_parts[1])

                    # ピーク長を5-8秒に制限（ヒートマップと同様の短さ）
                    peak_duration = end_sec - start_sec
                    if peak_duration < 3:
                        end_sec = start_sec + 5  # 最低5秒
                    elif peak_duration > 10:
                        end_sec = start_sec + 8  # 最大8秒

                    parsed_peaks.append({
                        "start": start_sec,
                        "end": end_sec,
                        "score": float(p.get("score", 0.7)),
                        "reason": p.get("reason", "")
                    })
                except (ValueError, KeyError):
                    continue

            # スコア順にソート
            parsed_peaks.sort(key=lambda x: x["score"], reverse=True)
            return parsed_peaks[:5]

        return []

    except Exception as e:
        print(f"LLM analysis error: {e}")
        return []


@app.post("/analyze-ai")
def analyze_with_ai(request: AnalyzeRequest):
    """Analyze YouTube video using AI (for videos without heatmap)

    Uses chunked transcription to handle long videos without API size limits.
    """
    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    client = get_groq_client()
    if not client:
        raise HTTPException(status_code=500, detail="Groq API key not configured")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")

            # 1. Download video
            print(f"Downloading video: {video_id}")
            ydl_opts = get_ydl_opts(video_path)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

            # ダウンロードされたファイルを探す
            import glob
            downloaded_files = glob.glob(os.path.join(tmpdir, "video.*"))
            if downloaded_files:
                video_path = downloaded_files[0]

            # 2. Transcribe with chunked processing (handles long videos)
            print("Starting chunked transcription...")
            segments = transcribe_audio_chunked(video_path, tmpdir, chunk_duration=600)

            if not segments:
                raise HTTPException(status_code=500, detail="Transcription failed")

            print(f"Transcription complete: {len(segments)} segments")

            # 3. Analyze with LLM
            peaks = analyze_peaks_with_llm(segments)

            if not peaks:
                raise HTTPException(status_code=404, detail="No peaks found by AI analysis")

            # フォーマットを/analyzeと同じ形式に
            result = []
            for p in peaks:
                start_min = int(p["start"] // 60)
                start_sec = int(p["start"] % 60)
                end_min = int(p["end"] // 60)
                end_sec = int(p["end"] % 60)

                result.append({
                    "time_range": f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}",
                    "score": p["score"],
                    "start": p["start"],
                    "end": p["end"],
                    "reason": p.get("reason", ""),
                    "source": "ai"
                })

            return {"peaks": result, "source": "ai"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"analyze-ai error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
            ydl_opts = get_ydl_opts(video_path)
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                # ダウンロードされたファイルを探す
                import glob
                downloaded_files = glob.glob(os.path.join(tmpdir, "video.*"))
                if downloaded_files:
                    video_path = downloaded_files[0]
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

            ydl_opts = get_ydl_opts(video_path)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

            # ダウンロードされたファイルを探す
            import glob
            downloaded_files = glob.glob(os.path.join(tmpdir, "video.*"))
            if downloaded_files:
                video_path = downloaded_files[0]

            # 2. Transcribe with chunked processing (handles long videos)
            jobs[job_id]["step"] = "transcribing"
            jobs[job_id]["progress"] = 30

            print(f"Starting chunked transcription for hook-first...")
            segments = transcribe_audio_chunked(video_path, tmpdir, chunk_duration=600)
            print(f"Transcription complete: {len(segments)} segments")

            # 3. Calculate hook-first structure
            jobs[job_id]["step"] = "analyzing"
            jobs[job_id]["progress"] = 50

            peak_start = request.peak_start
            peak_end = request.peak_end
            hook_duration = request.hook_duration
            context_duration = request.context_duration

            print(f"Input: peak_start={peak_start}, peak_end={peak_end}, hook_duration={hook_duration}, context_duration={context_duration}")

            # フックの開始位置（ピークの少し前から）
            hook_start = max(0, peak_end - hook_duration)

            # 文の区切りに合わせる
            hook_start = find_sentence_boundary(segments, hook_start, "before")
            hook_end = find_sentence_boundary(segments, peak_end, "after")

            print(f"After sentence boundary: hook_start={hook_start}, hook_end={hook_end}")

            # 実際のフック長を計算
            actual_hook_duration = hook_end - hook_start
            if actual_hook_duration < 2:
                hook_end = hook_start + hook_duration

            # コンテキストの開始位置（フックの前）
            context_start = max(0, hook_start - context_duration)
            context_start = find_sentence_boundary(segments, context_start, "before")
            context_end = hook_start

            # 出力時間の計算とログ
            total_duration = (hook_end - hook_start) * 2 + (context_end - context_start)
            print(f"Structure: hook={hook_start}-{hook_end} ({hook_end-hook_start}s), context={context_start}-{context_end} ({context_end-context_start}s)")
            print(f"Total duration: {total_duration}s")

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
