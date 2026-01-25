"""Data models for YouTube clip generation"""

from dataclasses import dataclass, field


@dataclass
class HeatmapSegment:
    """Heatmapの1セグメントを表すデータクラス"""

    start_ms: int  # 開始時間（ミリ秒）
    duration_ms: int  # 区間長（ミリ秒）
    intensity_score: float  # 盛り上がり度（0-1）

    @property
    def end_ms(self) -> int:
        """終了時間（ミリ秒）"""
        return self.start_ms + self.duration_ms

    @property
    def start_seconds(self) -> float:
        """開始時間（秒）"""
        return self.start_ms / 1000.0

    @property
    def end_seconds(self) -> float:
        """終了時間（秒）"""
        return self.end_ms / 1000.0

    @property
    def duration_seconds(self) -> float:
        """区間長（秒）"""
        return self.duration_ms / 1000.0


@dataclass
class ClipCandidate:
    """切り抜き候補を表すデータクラス"""

    start_ms: int  # 開始時間（ミリ秒）
    end_ms: int  # 終了時間（ミリ秒）
    average_intensity: float  # 平均盛り上がり度
    segments: list[HeatmapSegment] = field(default_factory=list)  # 含まれるセグメント

    @property
    def duration_ms(self) -> int:
        """区間長（ミリ秒）"""
        return self.end_ms - self.start_ms

    @property
    def start_seconds(self) -> float:
        """開始時間（秒）"""
        return self.start_ms / 1000.0

    @property
    def end_seconds(self) -> float:
        """終了時間（秒）"""
        return self.end_ms / 1000.0

    @property
    def duration_seconds(self) -> float:
        """区間長（秒）"""
        return self.duration_ms / 1000.0

    def format_time_range(self) -> str:
        """時間範囲を読みやすい形式で返す"""
        start = self._format_time(self.start_seconds)
        end = self._format_time(self.end_seconds)
        return f"{start} - {end}"

    @staticmethod
    def _format_time(seconds: float) -> str:
        """秒を MM:SS 形式に変換"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


@dataclass
class TranscriptSegment:
    """文字起こしの1セグメントを表すデータクラス"""

    start_ms: int  # 開始時間（ミリ秒）
    end_ms: int  # 終了時間（ミリ秒）
    text: str  # テキスト

    @property
    def start_seconds(self) -> float:
        """開始時間（秒）"""
        return self.start_ms / 1000.0

    @property
    def end_seconds(self) -> float:
        """終了時間（秒）"""
        return self.end_ms / 1000.0

    @property
    def duration_seconds(self) -> float:
        """区間長（秒）"""
        return (self.end_ms - self.start_ms) / 1000.0

    def is_sentence_end(self) -> bool:
        """文末かどうかを判定"""
        sentence_endings = ("。", "！", "？", ".", "!", "?", "…")
        return self.text.rstrip().endswith(sentence_endings)


@dataclass
class VideoMetadata:
    """動画のメタデータを表すデータクラス"""

    video_id: str
    title: str
    duration_seconds: float
    channel: str
    upload_date: str | None = None
    view_count: int | None = None
