"""
各役割ごとのプロンプトテンプレートをまとめるモジュール。
プロンプト調整はこのファイルだけ触れば済むようにする。
"""

from __future__ import annotations

from typing import List

from schemas import FrameAnalysis  # 軽い依存なので reload は不要でもOK


def build_vision_caption_prompt() -> str:
    """
    画像1枚に対して caption + tags + scores を JSON で返すよう指示するプロンプト。
    """
    return (
        "以下の画像を見て、次の情報を日本語で JSON 形式で出力してください。\n"
        "- caption: 画像の内容を1文で説明\n"
        "- tags: 画像の特徴を表す短いタグの配列（例: ['室内', 'おもちゃ', '笑顔']）\n"
        "- scores: 'cuteness', 'interesting', 'representative' など0〜1のスコア\n"
        "出力は必ず JSON オブジェクトのみとし、余計な文章は一切書かないでください。"
    )


def build_diary_prompt(frame_analyses: List[FrameAnalysis], max_chars: int, language: str = "ja") -> str:
    """
    フレームの caption 群から 1日のミニ日記を書かせるプロンプト。
    """
    lines = [f"- {fa.caption}" for fa in frame_analyses]
    joined = "\n".join(lines)
    if language == "ja":
        instr = (
            "上記の箇条書きは、ある1日の映像を数秒ごとに要約した文です。\n"
            f"これらをもとに、{max_chars}文字以内の日本語で「今日の様子」を温かいトーンでまとめてください。\n"
            "時系列の流れがなんとなく分かるようにしつつ、細かすぎる描写は省いて大丈夫です。"
        )
    else:
        instr = (
            "The bullet points above are captions describing moments in a single day.\n"
            f"Write a warm, short diary-style summary in {language} within {max_chars} characters."
        )

    return joined + "\n\n" + instr
