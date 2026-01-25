"""YouTube切り抜き動画生成 - Streamlit UI"""

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import streamlit as st

from src.services.heatmap import HeatmapError, HeatmapService
from src.services.video import VideoError, VideoService
from src.services.transcript import TranscriptError, TranscriptService
from src.services.llm_clipper import LLMClipper, LLMClipperError
from src.utils.ffmpeg import FFmpegError, FFmpegWrapper
from src.models.clip import ClipCandidate

# ===== 内部設定 =====
THRESHOLD = 0.5          # 盛り上がり閾値
TOP_N = 10               # 生成するクリップ数
MAX_DURATION = 90        # 最大クリップ長（秒）
MODE = "hook-first"      # 編集モード
HOOK_DURATION = 5        # フック長（秒）
CONTEXT_DURATION = 40    # 文脈長（秒）
USE_LLM = True           # LLM最適化
LLM_PROVIDER = "groq"    # LLMプロバイダー
ADD_SUBTITLE = False     # 字幕追加
CONVERT_VERTICAL = False # 縦型変換
# ====================


# ===== ヘルパー関数 =====

def _limit_duration(candidate: ClipCandidate, max_dur: int) -> ClipCandidate:
    """候補の長さを制限"""
    max_duration_ms = max_dur * 1000
    if candidate.duration_ms <= max_duration_ms:
        return candidate
    return ClipCandidate(
        start_ms=candidate.start_ms,
        end_ms=candidate.start_ms + max_duration_ms,
        average_intensity=candidate.average_intensity,
        segments=[s for s in candidate.segments if s.start_ms < candidate.start_ms + max_duration_ms],
    )


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    """動画から音声を抽出"""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _reencode_video(input_path: Path, output_path: Path) -> None:
    """動画を再エンコード"""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _generate_hook_first_clip(
    video_path: Path,
    tmppath: Path,
    candidate: ClipCandidate,
    hook_dur: int,
    context_dur: int,
    llm_clipper=None,
    full_transcript=None,
    clip_idx: int = 1,
) -> Path:
    """hook-firstモードでクリップを生成"""
    unique_id = uuid.uuid4().hex[:8]

    peak_start = candidate.start_seconds
    peak_end = candidate.end_seconds

    # LLM提案がある場合
    llm_suggestion = None
    if llm_clipper and full_transcript:
        try:
            llm_suggestion = llm_clipper.analyze_for_hook_first(
                full_transcript, candidate, context_dur
            )
        except:
            pass

    if llm_suggestion and "hook" in llm_suggestion:
        hook_start = llm_suggestion["hook"].start_seconds
        hook_end = llm_suggestion["hook"].end_seconds
        if hook_end - hook_start < 3:
            hook_end = hook_start + hook_dur
    else:
        hook_start = peak_start
        hook_end = min(peak_start + hook_dur, peak_end)

    context_start = max(0, hook_start - context_dur)
    context_end = hook_start
    continuation_start = hook_end
    continuation_end = min(hook_end + 30, peak_end + 10)

    # フック
    hook_path = tmppath / f"hook_{unique_id}.mp4"
    FFmpegWrapper.cut_video(video_path, hook_path, hook_start, hook_end)

    parts = [hook_path]

    # 文脈
    if context_end > context_start + 1:
        context_path = tmppath / f"context_{unique_id}.mp4"
        FFmpegWrapper.cut_video(video_path, context_path, context_start, context_end)
        parts.append(context_path)

    # 続き
    if continuation_end > continuation_start + 1:
        cont_path = tmppath / f"cont_{unique_id}.mp4"
        FFmpegWrapper.cut_video(video_path, cont_path, continuation_start, continuation_end)
        parts.append(cont_path)

    # エンコード & 結合
    encoded = []
    for i, p in enumerate(parts):
        enc_path = tmppath / f"enc_{unique_id}_{i}.mp4"
        _reencode_video(p, enc_path)
        encoded.append(enc_path)

    combined = tmppath / f"combined_{unique_id}.mp4"
    FFmpegWrapper.concatenate_videos(encoded, combined)

    return combined


# ===== Streamlit UI =====

# ページ設定
st.set_page_config(
    page_title="YouTube切り抜き動画生成",
    page_icon="🎬",
    layout="wide",
)

# タイトル
st.title("🎬 YouTube切り抜き動画生成")
st.markdown("YouTube動画の盛り上がりポイントを自動検出し、ショート動画を生成します")

# YouTube URL入力
url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

# 生成ボタン
if st.button("🚀 クリップを生成", type="primary", use_container_width=True):
    if not url:
        st.error("YouTube URLを入力してください")
    elif not FFmpegWrapper.check_installed():
        st.error("FFmpegがインストールされていません")
    else:
        try:
            with st.spinner("処理中..."):
                progress = st.progress(0)
                status = st.empty()

                # 1. Heatmap取得
                status.text("📊 Heatmapを取得中...")
                progress.progress(10)

                heatmap_service = HeatmapService(threshold=THRESHOLD)
                segments = heatmap_service.fetch_heatmap(url)

                if not segments:
                    st.warning("Heatmapデータが見つかりませんでした。この動画は対応していない可能性があります。")
                    st.stop()

                # 2. ピーク検出
                status.text("🎯 盛り上がりポイントを検出中...")
                progress.progress(20)

                candidates = heatmap_service.detect_peaks(segments)

                if not candidates:
                    st.warning("盛り上がりポイントが見つかりませんでした。")
                    st.stop()

                # 候補表示
                st.subheader(f"🔥 検出された盛り上がりポイント（上位{min(TOP_N, len(candidates))}件）")
                for i, c in enumerate(candidates[:TOP_N], 1):
                    st.write(f"{i}. {c.format_time_range()} (スコア: {c.average_intensity:.2f})")

                # 3. 動画ダウンロード
                status.text("📥 動画をダウンロード中...")
                progress.progress(30)

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmppath = Path(tmpdir)
                    video_service = VideoService(output_dir=tmppath)
                    video_path = video_service.download_video(url)

                    # LLM初期化
                    llm_clipper = None
                    full_transcript = None

                    if USE_LLM:
                        status.text("🤖 LLM用に音声を文字起こし中...")
                        progress.progress(40)

                        try:
                            llm_clipper = LLMClipper(provider=LLM_PROVIDER)

                            audio_path = tmppath / "full_audio.wav"
                            _extract_audio(video_path, audio_path)

                            transcript_service = TranscriptService(model_name="base", language="ja")
                            full_transcript = transcript_service.transcribe(audio_path)
                        except Exception as e:
                            st.warning(f"LLM初期化エラー: {e}")

                    # クリップ生成
                    selected = candidates[:TOP_N]
                    selected = [_limit_duration(c, MAX_DURATION) if c.duration_seconds > MAX_DURATION else c for c in selected]

                    output_files = []

                    for idx, candidate in enumerate(selected, 1):
                        status.text(f"✂️ クリップ {idx}/{len(selected)} を生成中...")
                        progress.progress(40 + int(50 * idx / len(selected)))

                        if MODE == "hook-first":
                            clip_path = _generate_hook_first_clip(
                                video_path, tmppath, candidate,
                                HOOK_DURATION, CONTEXT_DURATION,
                                llm_clipper, full_transcript,
                                idx
                            )
                        else:
                            clip_path = tmppath / f"clip_{idx}.mp4"
                            FFmpegWrapper.cut_video(
                                video_path, clip_path,
                                candidate.start_seconds, candidate.end_seconds
                            )

                        current = clip_path

                        # 縦型変換
                        if CONVERT_VERTICAL:
                            vert_path = tmppath / f"vert_{idx}.mp4"
                            FFmpegWrapper.convert_to_vertical(current, vert_path)
                            current = vert_path

                        # 字幕
                        if ADD_SUBTITLE:
                            audio_p = tmppath / f"audio_{idx}.wav"
                            _extract_audio(current, audio_p)
                            ts = TranscriptService(model_name="base", language="ja")
                            segs = ts.transcribe(audio_p)
                            if segs:
                                srt_p = tmppath / f"sub_{idx}.srt"
                                ts.generate_srt(segs, srt_p)
                                sub_path = tmppath / f"subtitled_{idx}.mp4"
                                FFmpegWrapper.burn_subtitles(current, sub_path, srt_p)
                                current = sub_path

                        # 出力ディレクトリに保存
                        output_dir = Path("output")
                        output_dir.mkdir(exist_ok=True)

                        match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
                        video_id = match.group(1) if match else "output"

                        final_path = output_dir / f"clip_{video_id}_{idx}.mp4"

                        shutil.copy2(current, final_path)
                        output_files.append(final_path)

                    progress.progress(100)
                    status.text("✅ 完了!")

                # 結果表示
                st.success(f"🎉 {len(output_files)}本のクリップを生成しました！")

                st.subheader("📹 生成されたクリップ")

                tabs = st.tabs([f"クリップ {i+1}" for i in range(len(output_files))])

                for idx, (tab, path) in enumerate(zip(tabs, output_files)):
                    with tab:
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.video(str(path))

                        with col2:
                            st.code(path.name)
                            size_mb = path.stat().st_size / (1024 * 1024)
                            st.write(f"**{size_mb:.1f} MB**")

                            with open(path, "rb") as f:
                                st.download_button(
                                    f"⬇️ ダウンロード",
                                    f,
                                    file_name=path.name,
                                    mime="video/mp4",
                                    key=f"dl_{idx}",
                                    use_container_width=True
                                )

                # ファイルパス一覧
                st.subheader("📁 ファイルパス")
                st.code("\n".join([str(p) for p in output_files]))

        except HeatmapError as e:
            st.error(f"Heatmapエラー: {e}")
        except VideoError as e:
            st.error(f"動画エラー: {e}")
        except FFmpegError as e:
            st.error(f"FFmpegエラー: {e}")
        except Exception as e:
            st.error(f"エラー: {e}")
            import traceback
            st.code(traceback.format_exc())
