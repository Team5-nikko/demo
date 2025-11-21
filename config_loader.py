"""
設定ファイル(settings.yaml, models.yaml)を読み込んで Python オブジェクトとして提供するモジュール。
Colab での利用を想定し、ファイルが無い場合はデフォルト値で動くようにしている。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import os

import yaml


CONFIG_DIR = Path("config")
SETTINGS_PATH = CONFIG_DIR / "settings.yaml"
MODELS_PATH = CONFIG_DIR / "models.yaml"


@dataclass
class Settings:
    # データ保存ルート
    data_root: Path = Path("outputs")

    # 動画関連
    frame_interval_sec: float = 5.0

    # ベストショット
    max_bestshots: int = 2

    # 日記関連
    diary_max_chars: int = 500
    diary_language: str = "ja"

    # ログなど
    log_level: str = "INFO"


@dataclass
class ModelSettings:
    """
    役割ごとのモデル設定を保持する。
    例:
    {
      "vision_caption": {"model_name": "local-llava", "type": "vision", ...},
      "diary_writer":   {"model_name": "local-llama3-8b", "type": "text", ...}
    }
    """
    roles: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings() -> Settings:
    raw = _safe_load_yaml(SETTINGS_PATH)
    settings = Settings()

    # 細かい項目は必要に応じて追加
    if "data_root" in raw:
        settings.data_root = Path(raw["data_root"])

    if "frame_interval_sec" in raw:
        settings.frame_interval_sec = float(raw["frame_interval_sec"])

    if "max_bestshots" in raw:
        settings.max_bestshots = int(raw["max_bestshots"])

    diary = raw.get("diary", {})
    if "max_chars" in diary:
        settings.diary_max_chars = int(diary["max_chars"])
    if "language" in diary:
        settings.diary_language = str(diary["language"])

    if "logging" in raw and "level" in raw["logging"]:
        settings.log_level = str(raw["logging"]["level"])

    return settings


def load_model_settings() -> ModelSettings:
    raw = _safe_load_yaml(MODELS_PATH)
    return ModelSettings(roles=raw or {})


# グローバル設定（import 時点で読み込んでおく）
SETTINGS = load_settings()
MODEL_SETTINGS = load_model_settings()
