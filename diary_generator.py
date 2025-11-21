"""
FrameAnalysis の JSONL から1日のミニ日記テキストを生成し、
Markdown ファイルとして保存するモジュール。

backend:
  - gemini: Gemini Text モデルで日記生成
  - dummy : caption 群から簡易な日記文を組み立てる
"""

from __future__ import annotations

from typing import List

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

from paths import get_analysis_path, get_diary_path
from schemas import FrameAnalysis
from jsonl_io import read_jsonl_as_dataclasses
from config_loader import SETTINGS
from model_loader import load_model_for_role
from prompt_templates import build_diary_prompt


def _call_text_model_gemini(model_info, prompt: str) -> str:
    """
    Gemini Text モデルを呼び出して日記テキストを生成する。
    """
    client = model_info["client"]
    model_name = model_info["model_name"]

    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    text = resp.text or ""
    # 文字数制限に合わせて切り詰め
    max_chars = SETTINGS.diary_max_chars
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text.strip()


def _call_text_model_dummy(prompt: str) -> str:
    """
    LLM を使わないダミー実装。
    prompt 内の "- " で始まる行を抜き出して簡単な日記を組み立てる。
    """
    lines = prompt.splitlines()
    captions = [line[2:].strip() for line in lines if line.startswith("- ")]
    n = len(captions)

    if n == 0:
        diary = "【ダミー日記】\n今日は短い記録しかありませんでした。"
    else:
        first = captions[0]
        last = captions[-1] if n > 1 else None

        diary_lines = []
        diary_lines.append("【ダミー日記】")
        diary_lines.append(f"今日は全部で {n} 個のシーンが記録されました。")
        diary_lines.append(f"最初のシーンは「{first}」でした。")
        if last and last != first:
            diary_lines.append(f"最後のシーンは「{last}」でした。")
        diary_lines.append("この文章はローカル LLM / Gemini の代わりにダミーコードで自動生成されています。")

        diary = "\n".join(diary_lines)

    max_chars = SETTINGS.diary_max_chars
    if len(diary) > max_chars:
        diary = diary[: max_chars - 3] + "..."
    return diary


def generate_diary(video_id: str) -> str:
    """
    analysis/{video_id}_analysis.jsonl を読み込み、1本の日記テキストを生成。
    diary/{video_id}_diary.md に保存してテキストを返す。
    """
    analysis_path = get_analysis_path(video_id)
    frames: List[FrameAnalysis] = read_jsonl_as_dataclasses(analysis_path, FrameAnalysis)
    if not frames:
        print(f"No analysis found: {analysis_path}")
        return ""

    prompt = build_diary_prompt(
        frame_analyses=frames,
        max_chars=SETTINGS.diary_max_chars,
        language=SETTINGS.diary_language,
    )

    model_info = load_model_for_role("diary_writer")
    backend = model_info["backend"]

    if backend == "gemini":
        diary_text = _call_text_model_gemini(model_info, prompt)
    else:
        diary_text = _call_text_model_dummy(prompt)

    out_path = get_diary_path(video_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(diary_text, encoding="utf-8")

    print(f"Wrote diary markdown: {out_path}")
    return diary_text
