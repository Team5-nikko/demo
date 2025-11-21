"""
データ処理結果（フレーム・マニフェスト）を Colab 上で目視確認するための補助モジュール。
"""

from __future__ import annotations

from typing import List

from pathlib import Path
import random

import matplotlib.pyplot as plt  # type: ignore
import cv2  # type: ignore

import importlib
import paths
import schemas
import jsonl_io

importlib.reload(paths)
importlib.reload(schemas)
importlib.reload(jsonl_io)

from paths import get_frames_dir, get_manifest_path
from schemas import FrameMeta
from jsonl_io import read_jsonl_as_dataclasses


def show_sample_frames(video_id: str, n: int = 5) -> None:
    """
    指定 video_id のフレームからランダムに n 枚を表示。
    """
    frames_dir = get_frames_dir(video_id)
    all_paths = list(frames_dir.glob("*.png"))
    if not all_paths:
        print("No frames found.")
        return

    sample_paths = random.sample(all_paths, min(n, len(all_paths)))

    for p in sample_paths:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.title(p.name)
        plt.show()


def summarize_manifest(video_id: str) -> None:
    """
    マニフェストから簡単な統計情報を表示。
    """
    manifest_path = get_manifest_path(video_id)
    frames: List[FrameMeta] = read_jsonl_as_dataclasses(manifest_path, FrameMeta)

    print(f"video_id: {video_id}")
    print(f"#frames: {len(frames)}")
    if not frames:
        return

    times = [f.time_sec for f in frames]
    print(f"time range: {min(times):.2f}s - {max(times):.2f}s")
    n_dark = sum(1 for f in frames if f.is_too_dark)
    n_blur = sum(1 for f in frames if f.is_blurry)
    print(f"too_dark: {n_dark}, blurry: {n_blur}")
