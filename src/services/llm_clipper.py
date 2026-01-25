"""LLMを使った知的なクリップ判定サービス"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from src.models.clip import ClipCandidate, TranscriptSegment


class LLMClipperError(Exception):
    """LLMクリッパーのエラー"""
    pass


@dataclass
class LLMClipSuggestion:
    """LLMが提案するクリップ"""
    start_seconds: float
    end_seconds: float
    reason: str
    score: float = 0.0


# 日本語対応プロンプト
CLIP_ANALYSIS_PROMPT = """あなたは動画編集のプロです。以下の字幕テキストを分析し、最も視聴者を惹きつける「盛り上がりポイント」を特定してください。

## 分析基準
1. **フック力**: 冒頭で視聴者の興味を引く内容か
2. **ストーリー性**: 話の流れが自然で理解しやすいか
3. **クライマックス**: 感情的なピーク、驚き、笑い、感動があるか
4. **整合性**: 途中で話が切れていないか、文脈が完結しているか

## 字幕データ
{srt_content}

## 出力形式
以下の形式で、最適なクリップ区間を1〜3個提案してください：

```
CLIP 1:
START: [開始秒数]
END: [終了秒数]
REASON: [この区間を選んだ理由]
SCORE: [0.0-1.0のスコア]

CLIP 2:
...
```

重要：
- 文の途中で切らないでください
- 話の流れが自然に始まり、自然に終わる区間を選んでください
- 視聴者が「何の話？」と混乱しない区間を選んでください
"""

HOOK_FIRST_PROMPT = """あなたは動画編集のプロです。以下の字幕テキストから、「hook-first」形式の動画を作成するための最適な構成を提案してください。

## hook-first形式とは
1. **フック（冒頭5秒）**: 最も衝撃的・興味を引く瞬間を最初に見せる
2. **文脈（30-40秒）**: フックに至るまでの背景・流れを見せる
3. **クライマックス**: フックの瞬間を含む盛り上がり全体を見せる

## 字幕データ
{srt_content}

## 現在の盛り上がり区間
開始: {peak_start}秒
終了: {peak_end}秒

## 出力形式
以下の形式で最適な構成を提案してください：

```
HOOK:
START: [フック開始秒数]
END: [フック終了秒数]
REASON: [この瞬間をフックに選んだ理由]

CONTEXT:
START: [文脈開始秒数]
END: [文脈終了秒数]
REASON: [この区間を文脈に選んだ理由]

CLIMAX:
START: [クライマックス開始秒数]
END: [クライマックス終了秒数]
REASON: [この区間をクライマックスに選んだ理由]
```

重要：
- フックは視聴者が「え！？何これ！？」と思う瞬間を選んでください
- 文脈は「なぜそうなったか」が分かる区間を選んでください
- 文の途中で切らないでください
"""


class LLMClipper:
    """LLMを使った知的なクリップ判定"""

    def __init__(
        self,
        provider: str = "openai",  # "openai", "anthropic", "gemini"
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.provider = provider
        self.api_key = api_key or self._get_api_key(provider)
        self.model = model or self._get_default_model(provider)

        if not self.api_key:
            raise LLMClipperError(
                f"API key not found for {provider}. "
                f"Set {self._get_env_var_name(provider)} environment variable."
            )

    def _get_api_key(self, provider: str) -> str | None:
        env_var = self._get_env_var_name(provider)
        return os.environ.get(env_var)

    def _get_env_var_name(self, provider: str) -> str:
        return {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
        }.get(provider, "OPENAI_API_KEY")

    def _get_default_model(self, provider: str) -> str:
        return {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-sonnet-20241022",
            "gemini": "gemini-2.0-flash",
            "groq": "llama-3.3-70b-versatile",
        }.get(provider, "gpt-4o-mini")

    def analyze_for_clips(
        self,
        transcript_segments: list[TranscriptSegment],
        video_duration: float,
    ) -> list[LLMClipSuggestion]:
        """字幕を分析して最適なクリップ区間を提案"""
        srt_content = self._segments_to_srt(transcript_segments)
        prompt = CLIP_ANALYSIS_PROMPT.format(srt_content=srt_content)

        response = self._call_llm(prompt)
        return self._parse_clip_suggestions(response)

    def analyze_for_hook_first(
        self,
        transcript_segments: list[TranscriptSegment],
        candidate: ClipCandidate,
        context_duration: int = 40,
    ) -> dict[str, LLMClipSuggestion]:
        """hook-first形式の最適な構成を提案"""
        # 候補区間 + 前後の文脈を含む字幕を抽出
        context_start = max(0, candidate.start_seconds - context_duration - 10)
        context_end = candidate.end_seconds + 10

        relevant_segments = [
            s for s in transcript_segments
            if s.start_seconds >= context_start and s.end_seconds <= context_end
        ]

        srt_content = self._segments_to_srt(relevant_segments)
        prompt = HOOK_FIRST_PROMPT.format(
            srt_content=srt_content,
            peak_start=candidate.start_seconds,
            peak_end=candidate.end_seconds,
        )

        response = self._call_llm(prompt)
        return self._parse_hook_first_response(response)

    def _segments_to_srt(self, segments: list[TranscriptSegment]) -> str:
        """TranscriptSegmentをSRT形式に変換"""
        lines = []
        for i, seg in enumerate(segments):
            start_time = self._seconds_to_srt_time(seg.start_seconds)
            end_time = self._seconds_to_srt_time(seg.end_seconds)
            lines.append(f"{i + 1}")
            lines.append(f"{start_time} --> {end_time}")
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """秒をSRT時間形式に変換"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _call_llm(self, prompt: str) -> str:
        """LLM APIを呼び出し"""
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "groq":
            return self._call_groq(prompt)
        else:
            raise LLMClipperError(f"Unknown provider: {self.provider}")

    def _call_openai(self, prompt: str) -> str:
        """OpenAI APIを呼び出し"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content
        except ImportError:
            raise LLMClipperError("openai package not installed. Run: pip install openai")
        except Exception as e:
            raise LLMClipperError(f"OpenAI API error: {e}")

    def _call_anthropic(self, prompt: str) -> str:
        """Anthropic APIを呼び出し"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except ImportError:
            raise LLMClipperError("anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            raise LLMClipperError(f"Anthropic API error: {e}")

    def _call_gemini(self, prompt: str) -> str:
        """Google Gemini APIを呼び出し"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            return response.text
        except ImportError:
            raise LLMClipperError("google-generativeai package not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise LLMClipperError(f"Gemini API error: {e}")

    def _call_groq(self, prompt: str) -> str:
        """Groq APIを呼び出し（OpenAI互換）"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content
        except ImportError:
            raise LLMClipperError("openai package not installed. Run: pip install openai")
        except Exception as e:
            raise LLMClipperError(f"Groq API error: {e}")

    def _parse_clip_suggestions(self, response: str) -> list[LLMClipSuggestion]:
        """LLMレスポンスからクリップ提案をパース"""
        suggestions = []

        # CLIP N: ... のパターンを探す
        clip_pattern = r"CLIP\s*\d+:.*?START:\s*([\d.]+).*?END:\s*([\d.]+).*?REASON:\s*(.+?)(?:SCORE:\s*([\d.]+))?"
        matches = re.findall(clip_pattern, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            try:
                start = float(match[0])
                end = float(match[1])
                reason = match[2].strip()
                score = float(match[3]) if match[3] else 0.8

                suggestions.append(LLMClipSuggestion(
                    start_seconds=start,
                    end_seconds=end,
                    reason=reason,
                    score=score,
                ))
            except (ValueError, IndexError):
                continue

        return suggestions

    def _parse_hook_first_response(self, response: str) -> dict[str, LLMClipSuggestion]:
        """hook-firstレスポンスをパース"""
        result = {}

        # 最小時間の要件
        min_durations = {
            "hook": 3.0,
            "context": 15.0,
            "climax": 10.0,
        }

        for section in ["HOOK", "CONTEXT", "CLIMAX"]:
            pattern = rf"{section}:.*?START:\s*([\d.]+).*?END:\s*([\d.]+).*?REASON:\s*(.+?)(?=(?:HOOK|CONTEXT|CLIMAX|$))"
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

            if match:
                try:
                    start = float(match.group(1))
                    end = float(match.group(2))
                    duration = end - start

                    # 最小時間チェック
                    min_dur = min_durations.get(section.lower(), 3.0)
                    if duration < min_dur:
                        # 最小時間を満たさない場合は終了時間を調整
                        end = start + min_dur

                    result[section.lower()] = LLMClipSuggestion(
                        start_seconds=start,
                        end_seconds=end,
                        reason=match.group(3).strip(),
                    )
                except (ValueError, IndexError):
                    pass

        # 必須セクションが全て揃っているか確認
        if "hook" not in result or "climax" not in result:
            return {}  # 不完全な場合は空を返す（デフォルト計算を使用）

        return result
