"""
マニフェスト(FrameMeta)を読み込み、Vision モデルで caption/tags/scores を生成して
FrameAnalysis の JSONL を出力するモジュール。

backend:
  - gemini: Gemini Vision (画像 + テキスト)
  - dummy : 画像ファイル名ベースのダミー
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from PIL import Image  # type: ignore

import importlib
import paths
import schemas
import jsonl_io
import config_loader
import model_loader
import prompt_templates

importlib.reload(paths)
importlib.reload(schemas)
importlib.reload(jsonl_io)
importlib.reload(config_loader)
importlib.reload(model_loader)
importlib.reload(prompt_templates)

from paths import get_manifest_path, get_analysis_path
from schemas import FrameMeta, FrameAnalysis
from jsonl_io import read_jsonl_as_dataclasses, write_jsonl
from model_loader import load_model_for_role
from prompt_templates import build_vision_caption_prompt


def _call_vision_model_gemini(model_info, image_path: str, prompt: str) -> dict:
    """
    Gemini API を使って画像1枚を解析し、JSONレスポンスを返す。

    model_info: model_loader.load_model_for_role("vision_caption") の戻り値
    image_path: 画像ファイルパス
    prompt    : 画像と一緒に渡すプロンプト（JSON で返してと指示する）
    """
    client = model_info["client"]
    model_name = model_info["model_name"]

    img = Image.open(image_path)

    # Gemini API: 画像 + テキストでコンテンツ生成 :contentReference[oaicite:1]{index=1}
    resp = client.models.generate_content(
        model=model_name,
        contents=[prompt, img],
    )

    text = resp.text or ""

    # モデルには「必ず JSON オブジェクトだけ出して」と頼んでいる前提で、
    # ここでは json.loads を試みる。失敗したら簡易フォールバック。
    try:
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("JSON is not an object")
    except Exception:
        # フォールバック：全部 caption に突っ込む
        result = {
            "caption": text.strip(),
            "tags": ["gemini", "fallback"],
            "scores": {"cuteness": 0.5, "interesting": 0.5, "representative": 0.5},
        }

    # 最低限のキーは保証しておく
    result.setdefault("caption", "")
    result.setdefault("tags", [])
    result.setdefault("scores", {})

    return result


def _call_vision_model_dummy(image_path: str) -> dict:
    """
    LLM を使わないダミー実装。
    """
    name = Path(image_path).name
    caption = f"{name} が映っているフレームです（ダミーキャプション）。"
    tags = ["dummy", "frame", "no-llm"]
    scores = {
        "cuteness": 0.5,
        "interesting": 0.5,
        "representative": 0.5,
    }
    return {
        "caption": caption,
        "tags": tags,
        "scores": scores,
    }


def run_captioning(video_id: str) -> List[FrameAnalysis]:
    """
    1. manifests/{video_id}_frames_manifest.jsonl を読む
    2. Vision モデル（Gemini or dummy）で FrameAnalysis を作る
    3. analysis/{video_id}_analysis.jsonl に保存
    """
    manifest_path = get_manifest_path(video_id)
    frames: List[FrameMeta] = read_jsonl_as_dataclasses(manifest_path, FrameMeta)
    if not frames:
        print(f"No frames found in manifest: {manifest_path}")
        return []

    model_info = load_model_for_role("vision_caption")
    backend = model_info["backend"]
    base_prompt = build_vision_caption_prompt()

    analyses: List[FrameAnalysis] = []

    for fm in frames:
        if backend == "gemini":
            result = _call_vision_model_gemini(model_info, fm.frame_path, base_prompt)
        else:
            result = _call_vision_model_dummy(fm.frame_path)

        caption = result.get("caption", "")
        tags = result.get("tags") or []
        scores = result.get("scores") or {}

        fa = FrameAnalysis(
            video_id=fm.video_id,
            frame_index=fm.frame_index,
            time_sec=fm.time_sec,
            frame_path=fm.frame_path,
            caption=caption,
            tags=tags,
            scores=scores,
        )
        analyses.append(fa)

    out_path = get_analysis_path(video_id)
    write_jsonl(out_path, analyses)
    print(f"Wrote analysis JSONL: {out_path}")
    return analyses
