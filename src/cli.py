"""YouTube切り抜き動画生成CLIツール"""

import tempfile
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.models.clip import ClipCandidate
from src.services.heatmap import HeatmapError, HeatmapService
from src.services.llm_clipper import LLMClipper, LLMClipperError
from src.services.transcript import TranscriptError, TranscriptService
from src.services.video import VideoError, VideoService
from src.utils.ffmpeg import FFmpegError, FFmpegWrapper

app = typer.Typer(
    name="youtube-cut",
    help="YouTube動画の盛り上がりポイントを自動検出し、ショート動画を生成するCLIツール",
    no_args_is_help=True,
)
console = Console()


@app.command()
def generate(
    url: Annotated[str, typer.Argument(help="YouTube動画のURL")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="出力ファイルパス"),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="盛り上がり判定の閾値（0-1）"),
    ] = 0.7,
    duration: Annotated[
        int,
        typer.Option("--duration", "-d", help="最大クリップ長（秒）"),
    ] = 60,
    mode: Annotated[
        str,
        typer.Option("--mode", help="編集モード: simple（通常）/ hook-first（フック→文脈→クライマックス）"),
    ] = "simple",
    hook_duration: Annotated[
        int,
        typer.Option("--hook-duration", help="フック部分の長さ（秒）[hook-firstモード用]"),
    ] = 5,
    context_duration: Annotated[
        int,
        typer.Option("--context-duration", help="文脈部分の長さ（秒）[hook-firstモード用]"),
    ] = 40,
    vertical: Annotated[
        bool,
        typer.Option("--vertical/--no-vertical", "-v", help="縦型（9:16）に変換"),
    ] = False,
    subtitle: Annotated[
        bool,
        typer.Option("--subtitle/--no-subtitle", "-s", help="字幕を焼き込み"),
    ] = True,
    whisper_model: Annotated[
        str,
        typer.Option("--whisper-model", "-m", help="Whisperモデル名"),
    ] = "base",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="詳細ログを出力"),
    ] = False,
    top: Annotated[
        int,
        typer.Option("--top", "-n", help="生成するクリップ数（検出された盛り上がりポイントの上位N件）"),
    ] = 1,
    llm: Annotated[
        bool,
        typer.Option("--llm/--no-llm", help="LLMを使って整合性のあるカットを行う"),
    ] = False,
    llm_provider: Annotated[
        str,
        typer.Option("--llm-provider", help="LLMプロバイダー: openai / anthropic / gemini"),
    ] = "openai",
    llm_model: Annotated[
        Optional[str],
        typer.Option("--llm-model", help="LLMモデル名（省略時はデフォルト）"),
    ] = None,
) -> None:
    """
    YouTube動画から盛り上がりポイントを検出し、ショート動画を生成
    """
    # FFmpegチェック
    if not FFmpegWrapper.check_installed():
        console.print("[red]エラー: FFmpegがインストールされていません[/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        try:
            # 1. Heatmap取得
            task = progress.add_task("Heatmapを取得中...", total=None)
            heatmap_service = HeatmapService(threshold=threshold)
            segments = heatmap_service.fetch_heatmap(url)
            progress.update(task, completed=True)

            if not segments:
                console.print("[yellow]警告: Heatmapデータが見つかりませんでした[/yellow]")
                raise typer.Exit(1)

            if verbose:
                console.print(f"[dim]取得したセグメント数: {len(segments)}[/dim]")

            # 2. ピーク検出
            task = progress.add_task("盛り上がりポイントを検出中...", total=None)
            candidates = heatmap_service.detect_peaks(segments)
            progress.update(task, completed=True)

            if not candidates:
                console.print("[yellow]警告: 盛り上がりポイントが見つかりませんでした[/yellow]")
                console.print("[dim]閾値を下げてみてください: --threshold 0.5[/dim]")
                raise typer.Exit(1)

            # 生成するクリップ数を決定
            num_clips = min(top, len(candidates))
            selected_candidates = candidates[:num_clips]

            # duration制限を適用
            selected_candidates = [
                _limit_duration(c, duration) if c.duration_seconds > duration else c
                for c in selected_candidates
            ]

            if verbose:
                _print_candidates(candidates[:max(5, num_clips)])

            console.print(f"\n[green]生成するクリップ数:[/green] {num_clips}")
            for i, candidate in enumerate(selected_candidates, 1):
                console.print(
                    f"  {i}. {candidate.format_time_range()} "
                    f"(スコア: {candidate.average_intensity:.2f})"
                )

            if mode == "hook-first":
                console.print(f"[cyan]モード:[/cyan] hook-first（フック→文脈→クライマックス）")

            if llm:
                console.print(f"[cyan]LLM:[/cyan] {llm_provider} ({llm_model or 'デフォルト'})")

            # 3. 動画ダウンロード
            task = progress.add_task("動画をダウンロード中...", total=None)
            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                video_service = VideoService(output_dir=tmppath)
                video_path = video_service.download_video(url)
                progress.update(task, completed=True)

                import shutil

                # LLMクリッパーの初期化（必要な場合）
                llm_clipper = None
                full_transcript = None
                if llm:
                    try:
                        llm_clipper = LLMClipper(
                            provider=llm_provider,
                            model=llm_model,
                        )
                        # 動画全体の文字起こしを取得
                        task = progress.add_task("LLM用に音声を文字起こし中...", total=None)
                        full_audio_path = tmppath / "full_audio.wav"
                        _extract_audio(video_path, full_audio_path)
                        transcript_service = TranscriptService(
                            model_name=whisper_model,
                            language="ja",
                        )
                        full_transcript = transcript_service.transcribe(full_audio_path)
                        progress.update(task, completed=True)
                        if verbose:
                            console.print(f"[dim]文字起こしセグメント数: {len(full_transcript)}[/dim]")
                    except LLMClipperError as e:
                        console.print(f"[yellow]LLM初期化エラー: {e}[/yellow]")
                        console.print("[yellow]LLMなしで続行します[/yellow]")
                        llm_clipper = None

                output_files = []
                video_id = _sanitize_filename(url)

                for clip_idx, candidate in enumerate(selected_candidates, 1):
                    clip_label = f"[{clip_idx}/{num_clips}]"

                    # LLMによる最適化（有効な場合）
                    llm_suggestion = None
                    if llm_clipper and full_transcript and mode == "hook-first":
                        task = progress.add_task(f"{clip_label} LLMで最適なカット位置を分析中...", total=None)
                        try:
                            llm_suggestion = llm_clipper.analyze_for_hook_first(
                                full_transcript,
                                candidate,
                                context_duration,
                            )
                            if verbose and llm_suggestion:
                                console.print(f"[dim]LLM提案:[/dim]")
                                for key, suggestion in llm_suggestion.items():
                                    console.print(f"[dim]  {key}: {suggestion.start_seconds:.1f}s - {suggestion.end_seconds:.1f}s[/dim]")
                                    console.print(f"[dim]    理由: {suggestion.reason[:50]}...[/dim]")
                        except LLMClipperError as e:
                            if verbose:
                                console.print(f"[dim]LLM分析スキップ: {e}[/dim]")
                        progress.update(task, completed=True)

                    if mode == "hook-first":
                        # hook-first モード: フック→文脈→クライマックス
                        current_path = _generate_hook_first(
                            progress=progress,
                            video_path=video_path,
                            tmppath=tmppath,
                            best_candidate=candidate,
                            hook_duration=hook_duration,
                            context_duration=context_duration,
                            verbose=verbose,
                            clip_label=clip_label,
                            llm_suggestion=llm_suggestion,
                        )
                    else:
                        # simple モード: 従来通り
                        task = progress.add_task(f"{clip_label} 動画を切り出し中...", total=None)
                        cut_path = tmppath / f"cut_{clip_idx}.mp4"
                        FFmpegWrapper.cut_video(
                            video_path,
                            cut_path,
                            candidate.start_seconds,
                            candidate.end_seconds,
                        )
                        progress.update(task, completed=True)
                        current_path = cut_path

                    # 5. 縦型変換（オプション）
                    if vertical:
                        task = progress.add_task(f"{clip_label} 縦型に変換中...", total=None)
                        vertical_path = tmppath / f"vertical_{clip_idx}.mp4"
                        FFmpegWrapper.convert_to_vertical(current_path, vertical_path)
                        current_path = vertical_path
                        progress.update(task, completed=True)

                    # 6. 字幕（オプション）
                    if subtitle:
                        task = progress.add_task(f"{clip_label} 字幕を生成中...", total=None)

                        # 音声抽出
                        audio_path = tmppath / f"audio_{clip_idx}.wav"
                        _extract_audio(current_path, audio_path)

                        # 文字起こし
                        transcript_service = TranscriptService(
                            model_name=whisper_model,
                            language="ja",
                        )
                        transcript_segments = transcript_service.transcribe(audio_path)

                        if transcript_segments:
                            # SRT生成
                            srt_path = tmppath / f"subtitle_{clip_idx}.srt"
                            transcript_service.generate_srt(transcript_segments, srt_path)

                            # 字幕焼き込み
                            subtitled_path = tmppath / f"subtitled_{clip_idx}.mp4"
                            FFmpegWrapper.burn_subtitles(current_path, subtitled_path, srt_path)
                            current_path = subtitled_path

                        progress.update(task, completed=True)

                    # 7. 出力
                    if output is not None:
                        if num_clips == 1:
                            clip_output = output
                        else:
                            # 複数クリップの場合はファイル名に番号を付ける
                            clip_output = output.parent / f"{output.stem}_{clip_idx}{output.suffix}"
                    else:
                        if num_clips == 1:
                            clip_output = Path(f"clip_{video_id}.mp4")
                        else:
                            clip_output = Path(f"clip_{video_id}_{clip_idx}.mp4")

                    clip_output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(current_path, clip_output)
                    output_files.append(clip_output)

            console.print(f"\n[green]完了![/green] 生成されたクリップ:")
            for f in output_files:
                console.print(f"  - {f}")

        except HeatmapError as e:
            console.print(f"[red]Heatmapエラー: {e}[/red]")
            raise typer.Exit(1)
        except VideoError as e:
            console.print(f"[red]動画エラー: {e}[/red]")
            raise typer.Exit(1)
        except TranscriptError as e:
            console.print(f"[red]文字起こしエラー: {e}[/red]")
            raise typer.Exit(1)
        except FFmpegError as e:
            console.print(f"[red]FFmpegエラー: {e}[/red]")
            raise typer.Exit(1)
        except LLMClipperError as e:
            console.print(f"[red]LLMエラー: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def analyze(
    url: Annotated[str, typer.Argument(help="YouTube動画のURL")],
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="盛り上がり判定の閾値（0-1）"),
    ] = 0.5,
    top: Annotated[
        int,
        typer.Option("--top", "-n", help="表示する上位件数"),
    ] = 10,
) -> None:
    """
    YouTube動画のHeatmapを分析し、盛り上がりポイントを表示
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        try:
            # Heatmap取得
            task = progress.add_task("Heatmapを取得中...", total=None)
            heatmap_service = HeatmapService(threshold=threshold)
            segments = heatmap_service.fetch_heatmap(url)
            progress.update(task, completed=True)

            if not segments:
                console.print("[yellow]警告: Heatmapデータが見つかりませんでした[/yellow]")
                raise typer.Exit(1)

            console.print(f"\n[bold]取得したセグメント数:[/bold] {len(segments)}")

            # ピーク検出
            candidates = heatmap_service.detect_peaks(segments)

            if not candidates:
                console.print("[yellow]閾値以上の盛り上がりポイントが見つかりませんでした[/yellow]")
                raise typer.Exit(1)

            console.print(f"[bold]検出した盛り上がりポイント数:[/bold] {len(candidates)}")
            console.print(f"[bold]閾値:[/bold] {threshold}")
            console.print()

            _print_candidates(candidates[:top])

        except HeatmapError as e:
            console.print(f"[red]エラー: {e}[/red]")
            raise typer.Exit(1)


def _print_candidates(candidates: list[ClipCandidate]) -> None:
    """候補をテーブル形式で表示"""
    table = Table(title="盛り上がりポイント")
    table.add_column("順位", justify="right", style="cyan")
    table.add_column("時間範囲", style="green")
    table.add_column("長さ", justify="right")
    table.add_column("スコア", justify="right", style="magenta")

    for i, candidate in enumerate(candidates, start=1):
        table.add_row(
            str(i),
            candidate.format_time_range(),
            f"{candidate.duration_seconds:.1f}秒",
            f"{candidate.average_intensity:.2f}",
        )

    console.print(table)


def _limit_duration(candidate: ClipCandidate, max_duration: int) -> ClipCandidate:
    """候補の長さを制限"""
    max_duration_ms = max_duration * 1000
    if candidate.duration_ms <= max_duration_ms:
        return candidate

    return ClipCandidate(
        start_ms=candidate.start_ms,
        end_ms=candidate.start_ms + max_duration_ms,
        average_intensity=candidate.average_intensity,
        segments=[
            s
            for s in candidate.segments
            if s.start_ms < candidate.start_ms + max_duration_ms
        ],
    )


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    """動画から音声を抽出"""
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _sanitize_filename(url: str) -> str:
    """URLからファイル名に使える文字列を生成"""
    import re

    # video_idを抽出
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    return "output"


def _generate_hook_first(
    progress: Progress,
    video_path: Path,
    tmppath: Path,
    best_candidate: ClipCandidate,
    hook_duration: int,
    context_duration: int,
    verbose: bool,
    clip_label: str = "",
    llm_suggestion: dict | None = None,
) -> Path:
    """
    hook-firstモードで動画を生成（重複なし）

    構成:
    1. フック（盛り上がりの最初の数秒）- 衝撃的な瞬間
    2. 文脈（盛り上がりの前の部分）- なぜそうなったか
    3. 続き（フックの後の部分）- フックを繰り返さず、その後の展開
    """
    peak_start = best_candidate.start_seconds
    peak_end = best_candidate.end_seconds

    # LLM提案がある場合はフック位置を参考にする
    if llm_suggestion and "hook" in llm_suggestion:
        hook_start = llm_suggestion["hook"].start_seconds
        hook_end = llm_suggestion["hook"].end_seconds
        # フックが短すぎる場合は調整
        if hook_end - hook_start < 3:
            hook_end = hook_start + hook_duration
    else:
        # デフォルト: 盛り上がりの最初の hook_duration 秒
        hook_start = peak_start
        hook_end = min(peak_start + hook_duration, peak_end)

    # 文脈: フックの前の部分（フックとは重複しない）
    context_start = max(0, hook_start - context_duration)
    context_end = hook_start  # フックの直前まで

    # 続き: フックの後の部分（フックを繰り返さない）
    continuation_start = hook_end  # フックの終わりから
    continuation_end = min(hook_end + 30, peak_end + 10)  # フック後30秒程度

    if verbose:
        console.print(f"[dim]フック: {hook_start:.1f}s - {hook_end:.1f}s ({hook_end - hook_start:.1f}秒)[/dim]")
        console.print(f"[dim]文脈: {context_start:.1f}s - {context_end:.1f}s ({context_end - context_start:.1f}秒)[/dim]")
        console.print(f"[dim]続き: {continuation_start:.1f}s - {continuation_end:.1f}s ({continuation_end - continuation_start:.1f}秒)[/dim]")

    # ユニークなサフィックスを生成（複数クリップ対応）
    import uuid
    unique_id = uuid.uuid4().hex[:8]

    # 1. フック部分を切り出し
    task = progress.add_task(f"{clip_label} フック部分を切り出し中...", total=None)
    hook_path = tmppath / f"hook_{unique_id}.mp4"
    FFmpegWrapper.cut_video(video_path, hook_path, hook_start, hook_end)
    progress.update(task, completed=True)

    # 2. 文脈部分を切り出し（文脈がある場合のみ）
    parts_to_concat = [hook_path]

    if context_end > context_start + 1:  # 1秒以上ある場合のみ
        task = progress.add_task(f"{clip_label} 文脈部分を切り出し中...", total=None)
        context_path = tmppath / f"context_{unique_id}.mp4"
        FFmpegWrapper.cut_video(video_path, context_path, context_start, context_end)
        progress.update(task, completed=True)
        parts_to_concat.append(context_path)

    # 3. 続き部分を切り出し（フックの後、重複なし）
    if continuation_end > continuation_start + 1:  # 1秒以上ある場合のみ
        task = progress.add_task(f"{clip_label} 続き部分を切り出し中...", total=None)
        continuation_path = tmppath / f"continuation_{unique_id}.mp4"
        FFmpegWrapper.cut_video(video_path, continuation_path, continuation_start, continuation_end)
        progress.update(task, completed=True)
        parts_to_concat.append(continuation_path)

    # 4. 全パーツを同じフォーマットにエンコード（結合のため）
    task = progress.add_task(f"{clip_label} 動画を結合中...", total=None)
    encoded_parts = []
    for i, part in enumerate(parts_to_concat):
        encoded_path = tmppath / f"encoded_{unique_id}_{i}.mp4"
        _reencode_video(part, encoded_path)
        encoded_parts.append(encoded_path)

    # 5. 結合
    combined_path = tmppath / f"combined_{unique_id}.mp4"
    FFmpegWrapper.concatenate_videos(encoded_parts, combined_path)
    progress.update(task, completed=True)

    return combined_path


def _reencode_video(input_path: Path, output_path: Path) -> None:
    """動画を再エンコード（結合用に統一フォーマットに）"""
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


if __name__ == "__main__":
    app()
