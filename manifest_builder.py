"""
FrameMeta のリストをマニフェスト JSONL として保存するモジュール。
LLM 側はこのマニフェストを起点にフレーム群を扱う。
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

from paths import get_manifest_path
from schemas import FrameMeta
from jsonl_io import write_jsonl


def build_manifest(video_id: str, frames: List[FrameMeta]) -> None:
    """
    FrameMeta のリストを JSONL として保存する。
    """
    manifest_path = get_manifest_path(video_id)
    write_jsonl(manifest_path, frames)
