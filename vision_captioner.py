
"""
マニフェスト(FrameMeta)を読み込み、Vision LLM で caption/tags/scores/位置情報 などを生成して
FrameAnalysis の JSONL を出力するモジュール。
"""

from __future__ import annotations

import json
import base64
from typing import List, Dict, Any, Optional

import importlib
import paths
import schemas
import jsonl_io
import config_loader
import model_loader
import vision_caption_prompt  # ★ ここからプロンプトを読み込む

importlib.reload(paths)
importlib.reload(schemas)
importlib.reload(jsonl_io)
importlib.reload(config_loader)
importlib.reload(model_loader)
importlib.reload(vision_caption_prompt)

from paths import get_manifest_path, get_analysis_path
from schemas import FrameMeta, FrameAnalysis
from jsonl_io import read_jsonl_as_dataclasses, write_jsonl
from config_loader import SETTINGS
from model_loader import load_model_for_role
from vision_caption_prompt import build_vision_caption_prompt


def _encode_image_base64(image_path: str) -> str:
    """画像ファイルを base64 文字列に変換するヘルパー。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_vision_model(model_info: Dict[str, Any], image_path: str, prompt: str) -> dict:
    """
    Vision モデル（SambaNova Llama-4-Maverick-17B-128E-Instruct など）を呼び出す。
    config/models.yaml の vision_caption セクションの設定に従って動作する。
    """
    backend = model_info["backend"]

    # -------- SambaNova バックエンド --------
    if backend == "sambanova":
        client = model_info["client"]
        model_name = model_info["model_name"]

        # 画像を base64 化
        image_b64 = _encode_image_base64(image_path)

        # 拡張子から MIME をざっくり判定（PNG 以外は JPEG 扱い）
        if image_path.lower().endswith(".png"):
            mime = "image/png"
        else:
            mime = "image/jpeg"

        # Vision + Text のマルチモーダル入力
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.2,
            top_p=0.9,
        )

        content = response.choices[0].message.content

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            text = "".join(texts)
        else:
            text = str(content)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # JSON になっていなかった場合でも、とりあえず caption にそのまま入れて返す
            return {
                "caption": text.strip(),
                "tags": [],
                "scores": {},
                "has_child": False,
                "num_children": 0,
                "main_subject": "",
                "bbox": None,
            }

    # -------- テスト用ダミー実装 --------
    elif backend == "dummy":
        return {
            "caption": "ダミー: 子どもが室内で遊んでいる様子です。",
            "tags": ["ダミー", "子ども"],
            "scores": {"cuteness": 0.5, "representative": 0.5},
            "has_child": True,
            "num_children": 1,
            "main_subject": "子ども",
            "bbox": [0.3, 0.3, 0.6, 0.8],
        }

    else:
        raise NotImplementedError(f"Unsupported backend for vision: {backend}")


def bbox_to_grid(
    bbox: List[float],
    num_rows: int = 10,
    num_cols: int = 10,
) -> tuple[int, int, str]:
    """
    bbox: [x_min, y_min, x_max, y_max]（0〜1正規化）を num_rows×num_cols のグリッド位置に変換。
    """
    if not bbox or len(bbox) < 4:
        raise ValueError(f"Invalid bbox: {bbox}")

    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    col = int(cx * num_cols)
    row = int(cy * num_rows)

    col = max(0, min(num_cols - 1, col))
    row = max(0, min(num_rows - 1, row))

    row_char = chr(ord("A") + row)
    label = f"{row_char}{col + 1}"
    return row, col, label


def _build_flags(
    has_child: bool,
    num_children: int,
    grid_label: Optional[str],
) -> Dict[str, bool]:
    flags: Dict[str, bool] = {}
    flags["child_present"] = bool(has_child)
    flags["multiple_children"] = num_children >= 2
    center_cells = {"E5", "E6", "F5", "F6"}
    flags["center_position"] = grid_label in center_cells if grid_label else False
    return flags

def run_captioning(video_id: str) -> List[FrameAnalysis]:
    """
    1. manifests/{video_id}_frames_manifest.jsonl を読む
    2. Vision LLM に投げて FrameAnalysis を作る
    3. analysis/{video_id}_analysis.jsonl に保存
    """
    manifest_path = get_manifest_path(video_id)
    frames: List[FrameMeta] = read_jsonl_as_dataclasses(manifest_path, FrameMeta)
    if not frames:
        print(f"No frames found in manifest: {manifest_path}")
        return []

    model_info = load_model_for_role("vision_caption")
    base_prompt = build_vision_caption_prompt()

    analyses: List[FrameAnalysis] = []

    for fm in frames:
        result: Dict[str, Any] = _call_vision_model(model_info, fm.frame_path, base_prompt)

        caption: str = result.get("caption", "")
        tags = result.get("tags") or []
        scores = result.get("scores") or {}

        has_child: bool = bool(result.get("has_child", False))
        num_children: int = int(result.get("num_children", 0))
        main_subject: str = result.get("main_subject", "") or ""

        bbox_raw = result.get("bbox")
        bbox: Optional[List[float]] = None
        grid_row: Optional[int] = None
        grid_col: Optional[int] = None
        grid_label: Optional[str] = None

        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) >= 4:
            try:
                bbox = [float(x) for x in bbox_raw[:4]]
                grid_row, grid_col, grid_label = bbox_to_grid(bbox)
            except Exception as e:
                print(f"[WARN] bbox_to_grid failed for frame {fm.frame_path}: {e}")
                bbox = None
                grid_row = grid_col = None
                grid_label = None

        flags = _build_flags(has_child, num_children, grid_label)

        fa = FrameAnalysis(
            video_id=fm.video_id,
            frame_index=fm.frame_index,
            time_sec=fm.time_sec,
            frame_path=fm.frame_path,
            caption=caption,
            tags=tags,
            scores=scores,
        )

        if hasattr(fa, "has_child"):
            setattr(fa, "has_child", has_child)
        if hasattr(fa, "num_children"):
            setattr(fa, "num_children", num_children)
        if hasattr(fa, "main_subject"):
            setattr(fa, "main_subject", main_subject)
        if hasattr(fa, "bbox"):
            setattr(fa, "bbox", bbox)
        if hasattr(fa, "grid_row"):
            setattr(fa, "grid_row", grid_row)
        if hasattr(fa, "grid_col"):
            setattr(fa, "grid_col", grid_col)
        if hasattr(fa, "grid_label"):
            setattr(fa, "grid_label", grid_label)
        if hasattr(fa, "flags"):
            setattr(fa, "flags", flags)

        if hasattr(fa, "extra"):
            current_extra = getattr(fa, "extra") or {}
            if not isinstance(current_extra, dict):
                current_extra = {}
            current_extra.update(
                {
                    "raw_vision_result": result,
                    "grid_info": {
                        "bbox": bbox,
                        "grid_row": grid_row,
                        "grid_col": grid_col,
                        "grid_label": grid_label,
                    },
                }
            )
            setattr(fa, "extra", current_extra)

        analyses.append(fa)

    out_path = get_analysis_path(video_id)
    write_jsonl(out_path, analyses)
    return analyses

