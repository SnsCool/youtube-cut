"""YouTube Heatmap (Most Replayed) データ取得・解析サービス"""

import json
import re
from typing import Any

import requests

from src.models.clip import ClipCandidate, HeatmapSegment


class HeatmapError(Exception):
    """Heatmap取得・解析時のエラー"""

    pass


class HeatmapService:
    """YouTube Heatmap データを取得・解析するサービス"""

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    INNERTUBE_API_URL = "https://www.youtube.com/youtubei/v1/next"

    def __init__(self, threshold: float = 0.7, merge_gap_ms: int = 5000) -> None:
        """
        Args:
            threshold: 盛り上がりと判定するintensityScoreの閾値（0-1）
            merge_gap_ms: この間隔以下の連続区間をマージする（ミリ秒）
        """
        self.threshold = threshold
        self.merge_gap_ms = merge_gap_ms

    def fetch_heatmap(self, video_url: str) -> list[HeatmapSegment]:
        """
        YouTube動画のHeatmapデータを取得

        Args:
            video_url: YouTube動画のURL

        Returns:
            HeatmapSegmentのリスト

        Raises:
            HeatmapError: 取得に失敗した場合
        """
        video_id = self._extract_video_id(video_url)
        if not video_id:
            raise HeatmapError(f"無効なYouTube URL: {video_url}")

        # InnerTube APIを使用してHeatmapデータを取得
        segments = self._fetch_via_innertube_api(video_id)

        if not segments:
            # フォールバック: HTML解析を試みる
            html = self._fetch_html(f"https://www.youtube.com/watch?v={video_id}")
            yt_initial_data = self._extract_yt_initial_data(html)
            if yt_initial_data:
                segments = self._parse_heatmap_markers(yt_initial_data)

        return segments

    def _fetch_via_innertube_api(self, video_id: str) -> list[HeatmapSegment]:
        """InnerTube APIを使用してHeatmapデータを取得"""
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
            "User-Agent": self.USER_AGENT,
        }

        try:
            response = requests.post(
                self.INNERTUBE_API_URL,
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            # frameworkUpdates -> entityBatchUpdate -> mutations からマーカーを取得
            mutations = (
                data.get("frameworkUpdates", {})
                .get("entityBatchUpdate", {})
                .get("mutations", [])
            )

            for mutation in mutations:
                markers_entity = mutation.get("payload", {}).get("macroMarkersListEntity")
                if markers_entity:
                    markers_list = markers_entity.get("markersList", {})
                    markers = markers_list.get("markers", [])
                    return self._parse_innertube_markers(markers)

            return []

        except requests.RequestException as e:
            raise HeatmapError(f"InnerTube API取得に失敗: {e}") from e
        except (KeyError, json.JSONDecodeError) as e:
            return []  # フォールバックへ

    def _parse_innertube_markers(self, markers: list[dict[str, Any]]) -> list[HeatmapSegment]:
        """InnerTube APIのマーカーをHeatmapSegmentに変換"""
        segments: list[HeatmapSegment] = []

        for marker in markers:
            try:
                segment = HeatmapSegment(
                    start_ms=int(marker.get("startMillis", 0)),
                    duration_ms=int(marker.get("durationMillis", 0)),
                    intensity_score=float(marker.get("intensityScoreNormalized", 0)),
                )
                segments.append(segment)
            except (ValueError, TypeError):
                continue

        segments.sort(key=lambda s: s.start_ms)
        return segments

    def detect_peaks(self, segments: list[HeatmapSegment]) -> list[ClipCandidate]:
        """
        Heatmapデータから盛り上がりポイントを検出

        Args:
            segments: HeatmapSegmentのリスト

        Returns:
            ClipCandidateのリスト（スコア降順）
        """
        if not segments:
            return []

        # 閾値以上のセグメントを抽出
        high_intensity = [s for s in segments if s.intensity_score >= self.threshold]

        if not high_intensity:
            return []

        # 連続するセグメントをマージ
        merged = self._merge_consecutive_segments(high_intensity)

        # スコア降順でソート
        merged.sort(key=lambda c: c.average_intensity, reverse=True)

        return merged

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

    def _fetch_html(self, url: str) -> str:
        """HTMLを取得"""
        headers = {"User-Agent": self.USER_AGENT}
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise HeatmapError(f"HTML取得に失敗: {e}") from e

    def _extract_yt_initial_data(self, html: str) -> dict[str, Any] | None:
        """HTMLからytInitialDataを抽出"""
        pattern = r"var ytInitialData = ({.*?});"
        match = re.search(pattern, html)
        if not match:
            # 別のパターンを試す
            pattern = r"ytInitialData\s*=\s*({.*?});"
            match = re.search(pattern, html)

        if match:
            try:
                return json.loads(match.group(1))  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                return None
        return None

    def _parse_heatmap_markers(self, data: dict[str, Any]) -> list[HeatmapSegment]:
        """ytInitialDataからheatmapマーカーを抽出"""
        segments: list[HeatmapSegment] = []

        # 再帰的にheatMarkerRendererを探索
        markers = self._find_heat_markers(data)

        for marker in markers:
            try:
                segment = HeatmapSegment(
                    start_ms=int(marker.get("timeRangeStartMillis", 0)),
                    duration_ms=int(marker.get("markerDurationMillis", 0)),
                    intensity_score=float(marker.get("heatMarkerIntensityScoreNormalized", 0)),
                )
                segments.append(segment)
            except (ValueError, TypeError):
                continue

        # 開始時間でソート
        segments.sort(key=lambda s: s.start_ms)
        return segments

    def _find_heat_markers(self, obj: Any, markers: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """再帰的にheatMarkerRendererを探索"""
        if markers is None:
            markers = []

        if isinstance(obj, dict):
            if "heatMarkerRenderer" in obj:
                markers.append(obj["heatMarkerRenderer"])
            for value in obj.values():
                self._find_heat_markers(value, markers)
        elif isinstance(obj, list):
            for item in obj:
                self._find_heat_markers(item, markers)

        return markers

    def _merge_consecutive_segments(
        self, segments: list[HeatmapSegment]
    ) -> list[ClipCandidate]:
        """連続するセグメントをマージしてClipCandidateを生成"""
        if not segments:
            return []

        # 開始時間でソート
        sorted_segments = sorted(segments, key=lambda s: s.start_ms)

        candidates: list[ClipCandidate] = []
        current_group: list[HeatmapSegment] = [sorted_segments[0]]

        for segment in sorted_segments[1:]:
            last = current_group[-1]
            gap = segment.start_ms - last.end_ms

            if gap <= self.merge_gap_ms:
                # マージ
                current_group.append(segment)
            else:
                # 新しいグループを開始
                candidates.append(self._create_candidate(current_group))
                current_group = [segment]

        # 最後のグループを追加
        candidates.append(self._create_candidate(current_group))

        return candidates

    def _create_candidate(self, segments: list[HeatmapSegment]) -> ClipCandidate:
        """セグメントグループからClipCandidateを生成"""
        start_ms = segments[0].start_ms
        end_ms = segments[-1].end_ms
        avg_intensity = sum(s.intensity_score for s in segments) / len(segments)

        return ClipCandidate(
            start_ms=start_ms,
            end_ms=end_ms,
            average_intensity=avg_intensity,
            segments=segments,
        )
