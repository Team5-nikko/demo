"""
Colab 上でアップロードされた動画ファイルを所定の場所に保存し、
video_id を発行するためのモジュール。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Union, IO

import importlib
import paths
importlib.reload(paths)  # Colab用

from paths import get_raw_video_path


def generate_video_id(prefix: str = "video") -> str:
    """
    時刻ベースの簡易 video_id を発行する。
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{now}_{prefix}"


def save_video(
    src: Union[str, Path, IO[bytes]],
    video_id: Optional[str] = None,
    suffix: str = ".mp4",
) -> str:
    """
    src: もとの動画ファイルパス、もしくは file-like object (binaries)
    戻り値: video_id
    """
    if video_id is None:
        video_id = generate_video_id()

    dst_path = get_raw_video_path(video_id, suffix=suffix)

    # src がパスか file-like かで分岐
    if isinstance(src, (str, Path)):
        src_path = Path(src)
        dst_path.write_bytes(src_path.read_bytes())
    else:
        # file-like
        data = src.read()
        dst_path.write_bytes(data)

    return video_id
