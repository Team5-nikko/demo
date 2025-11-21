"""
抽出したフレーム画像に対して、リサイズや簡単な画質チェックを行うモジュール。
（ブレ判定や暗さ判定など）
"""

from __future__ import annotations

from typing import List

import cv2  # type: ignore
import numpy as np  # type: ignore

import importlib
import schemas
importlib.reload(schemas)

from schemas import FrameMeta


def _is_too_dark(img, threshold: float = 40.0) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    return bool(mean_val < threshold)   # ★ ここをキャスト

def _is_blurry(img, threshold: float = 100.0) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return bool(fm < threshold)         # ★ 同じくキャスト


def preprocess_frames(
    frames: List[FrameMeta],
    resize_long_side: int = 640,
) -> List[FrameMeta]:
    """
    フレーム画像をリサイズし、暗さ/ブレのフラグを付与する。
    実際のモデル入力用の画像にもそのまま使える。
    """
    updated: List[FrameMeta] = []

    for meta in frames:
        img = cv2.imread(meta.frame_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = resize_long_side / max(h, w)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size)

        # 上書き保存（簡易）
        cv2.imwrite(meta.frame_path, img)

        meta.is_too_dark = _is_too_dark(img)
        meta.is_blurry = _is_blurry(img)
        updated.append(meta)

    return updated
