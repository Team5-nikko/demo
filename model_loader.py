"""
役割ごとに「どのバックエンドを使うか」を吸収するモジュール。

- backend: "gemini" -> Gemini API (google-genai)
- backend: "local"  -> 将来のローカル Llama 用（今は TODO）
- backend: "dummy"  -> ダミー実装（LLM無し）

実際の推論処理は vision_captioner / diary_generator 側で
model["backend"] を見て分岐する。
"""

from __future__ import annotations

from typing import Any, Dict
import os

import importlib
import config_loader
import secrets_helper

importlib.reload(config_loader)
importlib.reload(secrets_helper)

from config_loader import MODEL_SETTINGS
from secrets_helper import init_gemini_api_key

# Gemini SDK
try:
    from google import genai  # pip install -q -U google-genai
except ImportError:
    genai = None  # 後でエラーメッセージに使う


_MODEL_CACHE: Dict[str, Any] = {}
_GEMINI_CLIENT: Any = None


def get_model_config(role: str) -> Dict[str, Any]:
    """
    models.yaml から role に対応する設定を取得する。
    無ければ dummy backend を返す。
    """
    cfg = MODEL_SETTINGS.roles.get(role)
    if cfg is None:
        return {
            "backend": "dummy",
            "model_name": f"dummy-{role}",
            "type": "dummy",
        }
    return cfg


def _get_gemini_client() -> Any:
    global _GEMINI_CLIENT

    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT

    # ここで Colab / Cloud Run / ローカルのいずれかから GEMINI_API_KEY を初期化
    init_gemini_api_key()

    if genai is None:
        raise ImportError("google-genai がインストールされていません。")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が設定されていません。")

    _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


def load_model_for_role(role: str) -> Dict[str, Any]:
    """
    役割ごとの「モデル情報」を返す。
    戻り値は dict で、少なくとも以下を含む：
      - backend: "gemini" / "local" / "dummy"
      - model_name: str
      - client: バックエンド固有のオブジェクト（Geminiなら genai.Client）
    """
    if role in _MODEL_CACHE:
        return _MODEL_CACHE[role]

    cfg = get_model_config(role)
    backend = cfg.get("backend", "dummy")
    model_name = cfg.get("model_name", "")

    if backend == "gemini":
        client = _get_gemini_client()
        model = {
            "backend": "gemini",
            "client": client,
            "model_name": model_name,
            "role": role,
            "config": cfg,
        }

    elif backend == "local":
        # 将来、ローカル Llama をここでロードする想定
        # 今は未実装としておく（必要になったらここだけ実装すればよい）
        raise NotImplementedError(
            f"backend 'local' for role '{role}' is not implemented yet. "
            "ローカル Llama をつなぐときは model_loader.load_model_for_role を拡張してください。"
        )

    else:
        # dummy
        model = {
            "backend": "dummy",
            "client": None,
            "model_name": model_name or f"dummy-{role}",
            "role": role,
            "config": cfg,
        }

    _MODEL_CACHE[role] = model
    return model
