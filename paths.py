"""
ファイル/ディレクトリのパス・命名を一元管理するモジュール。
他のモジュールはここを経由してパスを取得し、文字列連結は原則禁止。
"""

from __future__ import annotations

from pathlib import Path

import importlib
import config_loader
importlib.reload(config_loader)  # Colab用: 編集後も最新を読む

from config_loader import SETTINGS


def get_data_root() -> Path:
    root = SETTINGS.data_root
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_raw_video_dir() -> Path:
    d = get_data_root() / "raw_videos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_raw_video_path(video_id: str, suffix: str = ".mp4") -> Path:
    return get_raw_video_dir() / f"{video_id}{suffix}"


def get_frames_dir(video_id: str) -> Path:
    d = get_data_root() / "frames" / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_frame_path(video_id: str, frame_index: int, ext: str = ".png") -> Path:
    return get_frames_dir(video_id) / f"{video_id}_f{frame_index:05d}{ext}"


def get_manifest_dir() -> Path:
    d = get_data_root() / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_manifest_path(video_id: str) -> Path:
    return get_manifest_dir() / f"{video_id}_frames_manifest.jsonl"


def get_analysis_dir() -> Path:
    d = get_data_root() / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_analysis_path(video_id: str) -> Path:
    return get_analysis_dir() / f"{video_id}_analysis.jsonl"


def get_bestshots_dir(video_id: str) -> Path:
    d = get_data_root() / "bestshots" / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_bestshot_image_path(video_id: str, rank: int, ext: str = ".png") -> Path:
    return get_bestshots_dir(video_id) / f"{video_id}_best_{rank:02d}{ext}"


def get_bestshot_meta_path(video_id: str) -> Path:
    return get_bestshots_dir(video_id) / f"{video_id}_bestshots.json"


def get_diary_dir() -> Path:
    d = get_data_root() / "diary"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_diary_path(video_id: str, ext: str = ".md") -> Path:
    return get_diary_dir() / f"{video_id}_diary{ext}"
