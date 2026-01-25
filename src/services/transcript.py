"""音声認識・文字起こしサービス"""

from pathlib import Path

from src.models.clip import TranscriptSegment


class TranscriptError(Exception):
    """文字起こし処理時のエラー"""

    pass


class TranscriptService:
    """Whisperを使用した文字起こしサービス"""

    def __init__(self, model_name: str = "base", language: str = "ja") -> None:
        """
        Args:
            model_name: Whisperモデル名（tiny, base, small, medium, large）
            language: 言語コード
        """
        self.model_name = model_name
        self.language = language
        self._model = None

    def _load_model(self) -> None:
        """Whisperモデルを遅延読み込み"""
        if self._model is not None:
            return

        try:
            import whisper

            self._model = whisper.load_model(self.model_name)
        except ImportError as e:
            raise TranscriptError(
                "whisperがインストールされていません: pip install openai-whisper"
            ) from e
        except Exception as e:
            raise TranscriptError(f"Whisperモデル読み込みエラー: {e}") from e

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        """
        音声ファイルを文字起こし

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            TranscriptSegmentのリスト
        """
        self._load_model()

        if not audio_path.exists():
            raise TranscriptError(f"音声ファイルが見つかりません: {audio_path}")

        try:
            result = self._model.transcribe(  # type: ignore[union-attr]
                str(audio_path),
                language=self.language,
                word_timestamps=True,
            )
        except Exception as e:
            raise TranscriptError(f"文字起こしエラー: {e}") from e

        segments: list[TranscriptSegment] = []
        for segment in result.get("segments", []):
            segments.append(
                TranscriptSegment(
                    start_ms=int(segment["start"] * 1000),
                    end_ms=int(segment["end"] * 1000),
                    text=segment["text"].strip(),
                )
            )

        return segments

    def find_sentence_boundary(
        self,
        segments: list[TranscriptSegment],
        target_ms: int,
        search_range_ms: int = 5000,
        prefer_after: bool = True,
    ) -> int:
        """
        指定時間付近の文末位置を探す

        Args:
            segments: TranscriptSegmentのリスト
            target_ms: 目標時間（ミリ秒）
            search_range_ms: 検索範囲（ミリ秒）
            prefer_after: Trueなら目標時間より後を優先

        Returns:
            最も近い文末の時間（ミリ秒）
        """
        if not segments:
            return target_ms

        # 文末のセグメントを抽出
        sentence_ends: list[int] = []
        for segment in segments:
            if segment.is_sentence_end():
                sentence_ends.append(segment.end_ms)

        if not sentence_ends:
            return target_ms

        # 検索範囲内の文末を探す
        candidates: list[tuple[int, int]] = []  # (距離, 時間)
        for end_ms in sentence_ends:
            distance = abs(end_ms - target_ms)
            if distance <= search_range_ms:
                # prefer_afterに応じて優先度を調整
                if prefer_after and end_ms >= target_ms:
                    distance -= 1  # 後ろを少し優先
                elif not prefer_after and end_ms <= target_ms:
                    distance -= 1  # 前を少し優先
                candidates.append((distance, end_ms))

        if not candidates:
            # 範囲内に文末がない場合は最も近いものを返す
            closest = min(sentence_ends, key=lambda x: abs(x - target_ms))
            return closest

        # 最も近い文末を返す
        candidates.sort()
        return candidates[0][1]

    def get_text_in_range(
        self,
        segments: list[TranscriptSegment],
        start_ms: int,
        end_ms: int,
    ) -> str:
        """
        指定範囲内のテキストを取得

        Args:
            segments: TranscriptSegmentのリスト
            start_ms: 開始時間（ミリ秒）
            end_ms: 終了時間（ミリ秒）

        Returns:
            連結されたテキスト
        """
        texts: list[str] = []
        for segment in segments:
            # セグメントが範囲と重なっているか確認
            if segment.end_ms > start_ms and segment.start_ms < end_ms:
                texts.append(segment.text)

        return " ".join(texts)

    def generate_srt(
        self,
        segments: list[TranscriptSegment],
        output_path: Path,
        offset_ms: int = 0,
    ) -> Path:
        """
        SRT形式の字幕ファイルを生成

        Args:
            segments: TranscriptSegmentのリスト
            output_path: 出力ファイルパス
            offset_ms: 時間オフセット（ミリ秒）

        Returns:
            出力ファイルパス
        """
        lines: list[str] = []
        for i, segment in enumerate(segments, start=1):
            start = self._ms_to_srt_time(segment.start_ms - offset_ms)
            end = self._ms_to_srt_time(segment.end_ms - offset_ms)
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(segment.text)
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    @staticmethod
    def _ms_to_srt_time(ms: int) -> str:
        """ミリ秒をSRT形式の時間に変換"""
        if ms < 0:
            ms = 0
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
