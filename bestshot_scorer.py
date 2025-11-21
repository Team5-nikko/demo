"""
FrameAnalysis の JSONL からベストショットを選定し、画像のコピーと
メタ情報 JSON を出力するモジュール。
ベースでは単純なスコアリングを使い、あとから LLM ベースに置き換え可能。
"""

from __future__ import annotations

import json
import shutil
from typing import List

import importlib
import paths
import schemas
import jsonl_io
import config_loader

importlib.reload(paths)
importlib.reload(schemas)
importlib.reload(jsonl_io)
importlib.reload(config_loader)

from paths import get_analysis_path, get_bestshot_image_path, get_bestshot_meta_path
from schemas import FrameAnalysis, BestShotMeta
from jsonl_io import read_jsonl_as_dataclasses
from config_loader import SETTINGS


def _compute_score(fa: FrameAnalysis) -> float:
    """
    ベースとなるスコアを計算する簡易ロジック。
    - すでに scores.cuteness があればそれを優先
    - 無ければ 0.5 固定など
    TODO: 後から洗練したロジックに差し替え可能。
    """
    if fa.scores:
        if "cuteness" in fa.scores:
            return float(fa.scores["cuteness"])
        # 他のスコアも適当に組み合わせられる
        return float(sum(fa.scores.values()) / max(len(fa.scores), 1))
    return 0.5


def select_bestshots(video_id: str) -> List[BestShotMeta]:
    """
    analysis/{video_id}_analysis.jsonl を読み込み、ベストショットを選定して
    bestshots/{video_id}_best_XX.png とメタ情報 JSON を出力する。
    """
    analysis_path = get_analysis_path(video_id)
    analyses: List[FrameAnalysis] = read_jsonl_as_dataclasses(analysis_path, FrameAnalysis)
    if not analyses:
        print(f"No analysis found: {analysis_path}")
        return []

    # スコア計算
    scored = [(fa, _compute_score(fa)) for fa in analyses]
    scored.sort(key=lambda x: x[1], reverse=True)

    n = min(SETTINGS.max_bestshots, len(scored))
    bestshots: List[BestShotMeta] = []

    for rank in range(n):
        fa, score = scored[rank]
        dst_img = get_bestshot_image_path(video_id, rank + 1)
        shutil.copy2(fa.frame_path, dst_img)

        meta = BestShotMeta(
            video_id=fa.video_id,
            frame_index=fa.frame_index,
            rank=rank + 1,
            score=float(score),
            frame_path=str(dst_img),
            caption=fa.caption,
        )
        bestshots.append(meta)

    # メタ情報を JSON で保存
    meta_path = get_bestshot_meta_path(video_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps([m.__dict__ for m in bestshots], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return bestshots
