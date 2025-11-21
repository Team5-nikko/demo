"""
動画から一定間隔でフレーム画像を抽出し、FrameMeta のリストとして返すモジュール。
OpenCV を想定。Colab では !pip install opencv-python が必要。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2  # type: ignore

import importlib
import config_loader
import paths
import schemas

importlib.reload(config_loader)
importlib.reload(paths)
importlib.reload(schemas)

from config_loader import SETTINGS
from paths import get_raw_video_path, get_frame_path
from schemas import FrameMeta


def extract_frames(video_id: str) -> List[FrameMeta]:
    """
    video_id に対応する動画ファイルから、設定された間隔ごとにフレームを抽出する。
    戻り値: FrameMeta のリスト
    """
    video_path = get_raw_video_path(video_id)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval_sec = SETTINGS.frame_interval_sec
    interval_frames = max(int(round(fps * interval_sec)), 1)

    frame_metas: List[FrameMeta] = []
    frame_index = 0
    saved_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % interval_frames == 0:
            time_sec = frame_index / fps
            out_path = get_frame_path(video_id, saved_index)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), frame)
            frame_metas.append(
                FrameMeta(
                    video_id=video_id,
                    frame_index=saved_index,
                    time_sec=time_sec,
                    frame_path=str(out_path),
                )
            )
            saved_index += 1

        frame_index += 1

    cap.release()
    return frame_metas
