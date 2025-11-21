"""
FrameAnalysis の時系列から異常っぽい区間を検出するモジュール。
ベースではルールベースで良く、あとから LLM を組み込んでもOK。
"""

from __future__ import annotations

from typing import List

import importlib
import paths
import schemas
import jsonl_io

importlib.reload(paths)
importlib.reload(schemas)
importlib.reload(jsonl_io)

from paths import get_analysis_path
from schemas import FrameAnalysis, AlertEvent
from jsonl_io import read_jsonl_as_dataclasses


def detect_simple_alerts(video_id: str) -> List[AlertEvent]:
    """
    非常にシンプルなルールベースの異常検知例。
    - caption や tags に「泣く」「転ぶ」などのキーワードがある場合にフラグ
    TODO: 実際のルールはチームで設計して差し替え。
    """
    analysis_path = get_analysis_path(video_id)
    frames: List[FrameAnalysis] = read_jsonl_as_dataclasses(analysis_path, FrameAnalysis)

    if not frames:
        print(f"No analysis found: {analysis_path}")
        return []

    keywords = ["泣", "転ぶ", "危ない", "暴れる"]
    events: List[AlertEvent] = []

    for fa in frames:
        if any(k in fa.caption for k in keywords):
            ev = AlertEvent(
                video_id=fa.video_id,
                start_time_sec=fa.time_sec,
                end_time_sec=fa.time_sec,
                level="warning",
                reason=f"keyword in caption: {fa.caption}",
                related_frames=[fa.frame_index],
            )
            events.append(ev)

    return events
