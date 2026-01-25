"""動画ダウンロード・処理サービス"""

import re
import subprocess
import tempfile
from pathlib import Path

from src.models.clip import VideoMetadata


class VideoError(Exception):
    """動画処理時のエラー"""

    pass


class VideoService:
    """YouTube動画のダウンロード・処理を行うサービス"""

    def __init__(self, output_dir: Path | None = None) -> None:
        """
        Args:
            output_dir: 出力ディレクトリ（Noneの場合は一時ディレクトリ）
        """
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "youtube-cut"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_video(
        self,
        video_url: str,
        output_path: Path | None = None,
        format_id: str | None = None,
    ) -> Path:
        """
        YouTube動画をダウンロード

        Args:
            video_url: YouTube動画のURL
            output_path: 出力ファイルパス（Noneの場合は自動生成）
            format_id: yt-dlpのフォーマット指定（Noneの場合は自動選択）

        Returns:
            ダウンロードしたファイルのパス

        Raises:
            VideoError: ダウンロードに失敗した場合
        """
        video_id = self._extract_video_id(video_url)
        if not video_id:
            raise VideoError(f"無効なYouTube URL: {video_url}")

        if output_path is None:
            output_path = self.output_dir / f"{video_id}.mp4"

        # 出力ファイルのテンプレート（拡張子は yt-dlp に任せる）
        output_template = str(output_path.with_suffix("")) + ".%(ext)s"

        cmd = [
            "yt-dlp",
            "-o",
            output_template,
            "--no-playlist",
            "--merge-output-format",
            "mp4",
        ]

        # フォーマット指定
        if format_id:
            cmd.extend(["-f", format_id])
        else:
            # 音声付き動画を優先、なければベスト
            cmd.extend(["-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"])

        cmd.append(video_url)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10分タイムアウト
            )
            if result.returncode != 0:
                # エラーメッセージを確認してリトライ
                if "empty" in result.stderr.lower() or "format" in result.stderr.lower():
                    # シンプルなフォーマットでリトライ
                    cmd_retry = [
                        "yt-dlp",
                        "-f", "18",  # 360p mp4 with audio
                        "-o", output_template,
                        "--no-playlist",
                        video_url,
                    ]
                    result = subprocess.run(
                        cmd_retry,
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                    if result.returncode != 0:
                        raise VideoError(f"yt-dlp エラー: {result.stderr}")
                else:
                    raise VideoError(f"yt-dlp エラー: {result.stderr}")
        except subprocess.TimeoutExpired as e:
            raise VideoError("ダウンロードがタイムアウトしました") from e
        except FileNotFoundError as e:
            raise VideoError("yt-dlpがインストールされていません") from e

        # 出力ファイルを探す
        possible_extensions = [".mp4", ".webm", ".mkv"]
        for ext in possible_extensions:
            alt_path = output_path.with_suffix(ext)
            if alt_path.exists() and alt_path.stat().st_size > 0:
                return alt_path

        # パターンマッチで探す
        for f in output_path.parent.glob(f"{output_path.stem}.*"):
            if f.suffix in possible_extensions and f.stat().st_size > 0:
                return f

        raise VideoError("ダウンロードしたファイルが見つかりません")

        return output_path

    def download_audio(
        self,
        video_url: str,
        output_path: Path | None = None,
    ) -> Path:
        """
        YouTube動画から音声のみをダウンロード

        Args:
            video_url: YouTube動画のURL
            output_path: 出力ファイルパス（Noneの場合は自動生成）

        Returns:
            ダウンロードしたファイルのパス
        """
        video_id = self._extract_video_id(video_url)
        if not video_id:
            raise VideoError(f"無効なYouTube URL: {video_url}")

        if output_path is None:
            output_path = self.output_dir / f"{video_id}.wav"

        cmd = [
            "yt-dlp",
            "-x",  # 音声のみ抽出
            "--audio-format",
            "wav",
            "-o",
            str(output_path.with_suffix("")),  # yt-dlpが拡張子を追加
            "--no-playlist",
            "--no-warnings",
            video_url,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                raise VideoError(f"yt-dlp エラー: {result.stderr}")
        except subprocess.TimeoutExpired as e:
            raise VideoError("ダウンロードがタイムアウトしました") from e

        # 拡張子を確認
        if output_path.exists():
            return output_path

        wav_path = output_path.with_suffix(".wav")
        if wav_path.exists():
            return wav_path

        raise VideoError("音声ファイルが見つかりません")

    def get_metadata(self, video_url: str) -> VideoMetadata:
        """
        動画のメタデータを取得

        Args:
            video_url: YouTube動画のURL

        Returns:
            VideoMetadata
        """
        video_id = self._extract_video_id(video_url)
        if not video_id:
            raise VideoError(f"無効なYouTube URL: {video_url}")

        cmd = [
            "yt-dlp",
            "--print",
            "%(title)s|||%(duration)s|||%(channel)s|||%(upload_date)s|||%(view_count)s",
            "--no-warnings",
            video_url,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                raise VideoError(f"メタデータ取得エラー: {result.stderr}")

            output = result.stdout.strip()
            parts = output.split("|||")

            return VideoMetadata(
                video_id=video_id,
                title=parts[0] if len(parts) > 0 else "Unknown",
                duration_seconds=float(parts[1]) if len(parts) > 1 and parts[1] else 0,
                channel=parts[2] if len(parts) > 2 else "Unknown",
                upload_date=parts[3] if len(parts) > 3 and parts[3] != "NA" else None,
                view_count=int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else None,
            )
        except subprocess.TimeoutExpired as e:
            raise VideoError("メタデータ取得がタイムアウトしました") from e

    def _extract_video_id(self, url: str) -> str | None:
        """URLからvideo_idを抽出"""
        patterns = [
            r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
            r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
