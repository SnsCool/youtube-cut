"""FFmpegラッパーユーティリティ"""

import subprocess
from pathlib import Path


class FFmpegError(Exception):
    """FFmpeg処理時のエラー"""

    pass


class FFmpegWrapper:
    """FFmpegコマンドのラッパー"""

    @staticmethod
    def check_installed() -> bool:
        """FFmpegがインストールされているか確認"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @staticmethod
    def cut_video(
        input_path: Path,
        output_path: Path,
        start_seconds: float,
        end_seconds: float,
    ) -> Path:
        """
        動画を指定範囲で切り出し

        Args:
            input_path: 入力ファイルパス
            output_path: 出力ファイルパス
            start_seconds: 開始時間（秒）
            end_seconds: 終了時間（秒）

        Returns:
            出力ファイルパス
        """
        duration = end_seconds - start_seconds

        cmd = [
            "ffmpeg",
            "-y",  # 上書き確認なし
            "-ss",
            str(start_seconds),
            "-i",
            str(input_path),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]

        FFmpegWrapper._run_command(cmd)
        return output_path

    @staticmethod
    def convert_to_vertical(
        input_path: Path,
        output_path: Path,
        width: int = 1080,
        height: int = 1920,
    ) -> Path:
        """
        横型動画を縦型（9:16）に変換

        Args:
            input_path: 入力ファイルパス
            output_path: 出力ファイルパス
            width: 出力幅（デフォルト: 1080）
            height: 出力高さ（デフォルト: 1920）

        Returns:
            出力ファイルパス
        """
        # 中央をクロップして縦型に変換
        # crop=出力幅:出力高さ:x座標:y座標
        # ih*9/16 で縦型の幅を計算、ih はそのまま
        vf = f"crop=ih*9/16:ih,scale={width}:{height}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(output_path),
        ]

        FFmpegWrapper._run_command(cmd)
        return output_path

    @staticmethod
    def burn_subtitles(
        input_path: Path,
        output_path: Path,
        subtitle_path: Path,
    ) -> Path:
        """
        字幕を動画に焼き込み

        Args:
            input_path: 入力ファイルパス
            output_path: 出力ファイルパス
            subtitle_path: 字幕ファイルパス（.srt または .ass）

        Returns:
            出力ファイルパス
        """
        # パスをエスケープ（Windows対応）
        escaped_sub_path = str(subtitle_path).replace("\\", "/").replace(":", "\\:")

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            f"subtitles='{escaped_sub_path}'",
            "-c:v",
            "libx264",
            "-c:a",
            "copy",
            str(output_path),
        ]

        FFmpegWrapper._run_command(cmd)
        return output_path

    @staticmethod
    def add_fade(
        input_path: Path,
        output_path: Path,
        fade_in_duration: float = 0.5,
        fade_out_duration: float = 0.5,
    ) -> Path:
        """
        フェードイン/アウトを追加

        Args:
            input_path: 入力ファイルパス
            output_path: 出力ファイルパス
            fade_in_duration: フェードイン時間（秒）
            fade_out_duration: フェードアウト時間（秒）

        Returns:
            出力ファイルパス
        """
        # 動画の長さを取得
        duration = FFmpegWrapper.get_duration(input_path)
        fade_out_start = duration - fade_out_duration

        vf = f"fade=t=in:st=0:d={fade_in_duration},fade=t=out:st={fade_out_start}:d={fade_out_duration}"
        af = f"afade=t=in:st=0:d={fade_in_duration},afade=t=out:st={fade_out_start}:d={fade_out_duration}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            vf,
            "-af",
            af,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(output_path),
        ]

        FFmpegWrapper._run_command(cmd)
        return output_path

    @staticmethod
    def get_duration(input_path: Path) -> float:
        """
        動画の長さを取得

        Args:
            input_path: 入力ファイルパス

        Returns:
            動画の長さ（秒）
        """
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
            raise FFmpegError(f"動画長さ取得エラー: {result.stderr}")
        except (ValueError, subprocess.TimeoutExpired) as e:
            raise FFmpegError(f"動画長さ取得エラー: {e}") from e

    @staticmethod
    def get_video_info(input_path: Path) -> dict[str, int]:
        """
        動画の情報を取得

        Args:
            input_path: 入力ファイルパス

        Returns:
            width, height を含む辞書
        """
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(input_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                width, height = result.stdout.strip().split("x")
                return {"width": int(width), "height": int(height)}
            raise FFmpegError(f"動画情報取得エラー: {result.stderr}")
        except (ValueError, subprocess.TimeoutExpired) as e:
            raise FFmpegError(f"動画情報取得エラー: {e}") from e

    @staticmethod
    def concatenate_videos(
        input_paths: list[Path],
        output_path: Path,
    ) -> Path:
        """
        複数の動画を結合

        Args:
            input_paths: 入力ファイルパスのリスト
            output_path: 出力ファイルパス

        Returns:
            出力ファイルパス
        """
        # concat用のファイルリストを作成
        list_file = output_path.parent / "concat_list.txt"
        with open(list_file, "w") as f:
            for path in input_paths:
                f.write(f"file '{path}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(output_path),
        ]

        try:
            FFmpegWrapper._run_command(cmd)
        finally:
            # 一時ファイルを削除
            if list_file.exists():
                list_file.unlink()

        return output_path

    @staticmethod
    def add_text_overlay(
        input_path: Path,
        output_path: Path,
        text: str,
        duration: float = 2.0,
        font_size: int = 48,
    ) -> Path:
        """
        動画にテキストオーバーレイを追加

        Args:
            input_path: 入力ファイルパス
            output_path: 出力ファイルパス
            text: 表示するテキスト
            duration: テキスト表示時間（秒）
            font_size: フォントサイズ

        Returns:
            出力ファイルパス
        """
        # drawtext フィルター
        vf = (
            f"drawtext=text='{text}':"
            f"fontsize={font_size}:"
            f"fontcolor=white:"
            f"borderw=3:"
            f"bordercolor=black:"
            f"x=(w-text_w)/2:"
            f"y=(h-text_h)/2:"
            f"enable='lt(t,{duration})'"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-c:a",
            "copy",
            str(output_path),
        ]

        FFmpegWrapper._run_command(cmd)
        return output_path

    @staticmethod
    def _run_command(cmd: list[str], timeout: int = 600) -> None:
        """FFmpegコマンドを実行"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise FFmpegError(f"FFmpegエラー: {result.stderr}")
        except FileNotFoundError as e:
            raise FFmpegError("FFmpegがインストールされていません") from e
        except subprocess.TimeoutExpired as e:
            raise FFmpegError("FFmpeg処理がタイムアウトしました") from e
