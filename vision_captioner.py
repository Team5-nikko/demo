import base64
from typing import Dict, Any

# すでに vision_captioner.py の先頭で model_loader を import している前提
# from model_loader import load_model_for_role  はそのままで OK


def _encode_image_base64(image_path: str) -> str:
    """画像ファイルを base64 文字列に変換するヘルパー。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_vision_model(model_info: Dict[str, Any], image_path: str, prompt: str) -> dict:
    """
    Vision モデル（SambaNova Llama-4-Maverick-17B-128E-Instruct など）を呼び出す。
    config/models.yaml の vision_caption セクションの設定に従って動作する。

    期待する戻り値（例）:
    {
      "caption": "...",
      "tags": [...],
      "scores": {...},
      "has_child": true/false,
      "num_children": 1,
      "main_subject": "...",
      "bbox": [0.3, 0.3, 0.6, 0.8]
    }
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

        # response.choices[0].message.content は
        # ・そのまま文字列
        # ・ content の配列
        # どちらかの可能性があるので両方対応
        content = response.choices[0].message.content

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # {"type": "text", "text": "..."} の配列を想定
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            text = "".join(texts)
        else:
            text = str(content)

        # プロンプト側で「必ず JSON を返して」と指示している前提なので、
        # まず JSON としてパースを試みる
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
