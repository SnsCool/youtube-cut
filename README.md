# youtube-cut

YouTube動画の盛り上がりポイントを自動検出し、ショート動画を生成するCLIツール

## 特徴

- **Heatmap分析**: YouTube Most Replayedデータを使用して盛り上がり箇所を特定
- **自動切り抜き**: 最も人気のある区間を自動で切り出し
- **ショート動画対応**: 9:16縦型フォーマットに自動変換
- **字幕生成**: Whisperによる高精度な文字起こしと字幕焼き込み

## インストール

```bash
pip install -e .
```

### 前提条件

- Python 3.11+
- FFmpeg
- yt-dlp

## 使い方

```bash
# 盛り上がりポイントを分析
youtube-cut analyze "https://www.youtube.com/watch?v=VIDEO_ID"

# ショート動画を生成
youtube-cut generate "https://www.youtube.com/watch?v=VIDEO_ID" -o output.mp4
```

## ライセンス

MIT
