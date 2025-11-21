"""
JSONL の1レコードや、各種メタデータの型定義をまとめたモジュール。
実際の JSONL はこのデータクラスを asdict したものを1行として保存する想定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FrameMeta:
    """
    LLM 解析前のフレーム情報。
    """
    video_id: str
    frame_index: int
    time_sec: float
    frame_path: str
    is_blurry: bool = False
    is_too_dark: bool = False


@dataclass
class FrameAnalysis:
    """
    LLM 解析後のフレーム情報（JSONLの1行に対応）。
    """
    video_id: str
    frame_index: int
    time_sec: float
    frame_path: str

    caption: str
    tags: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    # ここから追加
    has_child: bool = False          # 子どもが写っているか
    num_children: int = 0            # 推定人数
    main_subject: str = ""           # "boy" / "girl" / "teacher" など

    # バウンディングボックス（画像サイズで正規化0〜1）
    bbox: Optional[List[float]] = None   # [x_min, y_min, x_max, y_max]

    # 10×10グリッド上の位置
    grid_row: Optional[int] = None       # 0〜9
    grid_col: Optional[int] = None       # 0〜9
    grid_label: Optional[str] = None     # "A7" のような表記

    flags: Dict[str, bool] = field(default_factory=dict)

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BestShotMeta:
    """
    ベストショットとして選ばれたフレームのメタ情報。
    """
    video_id: str
    frame_index: int
    rank: int
    score: float
    frame_path: str
    caption: str


@dataclass
class AlertEvent:
    """
    異常検知用のイベント。
    """
    video_id: str
    start_time_sec: float
    end_time_sec: float
    level: str        # "info" / "warning" / "critical" など
    reason: str
    related_frames: List[int] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
