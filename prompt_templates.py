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
    Llama4向けに最適化された高品質なプロンプト。
    """
    return (
        "あなたは保育園で撮影された写真を解析する専門家です。"
        "以下の画像を注意深く観察し、指定された形式でJSONオブジェクトのみを出力してください。\n\n"
        
        "【出力形式】\n"
        "以下のJSON構造を厳密に守ってください。説明文やコメントは一切含めないでください。\n\n"
        
        "{\n"
        '  "caption": "画像の内容を1文で簡潔に説明してください。主語と述語を明確にし、具体的な行動や状況を含めてください。",\n'
        '  "tags": ["タグ1", "タグ2", "タグ3"],\n'
        '  "scores": {\n'
        '    "cuteness": 0.0-1.0,\n'
        '    "interesting": 0.0-1.0,\n'
        '    "representative": 0.0-1.0\n'
        "  },\n"
        '  "has_child": true/false,\n'
        '  "num_children": 0以上の整数,\n'
        '  "main_subject": "主な被写体（例: 男の子、女の子、先生、複数の子どもなど）",\n'
        '  "bbox": [x_min, y_min, x_max, y_max]\n'
        "}\n\n"
        
        "【各項目の詳細説明】\n"
        "1. caption: 画像に写っている内容を1文で説明。誰が、どこで、何をしているかを明確に。\n"
        "   例: \"室内の保育室で、男の子がブロックで遊んでいる。\"\n"
        "   例: \"屋外の園庭で、複数の子どもが滑り台で遊んでいる。\"\n\n"
        
        "2. tags: 画像の特徴を表す短いタグを配列で。以下のカテゴリから関連するものを選んでください。\n"
        "   - 場所: 室内、屋外、保育室、園庭、廊下、トイレ、給食室など\n"
        "   - 活動: 遊び、食事、お昼寝、制作、運動、読み聞かせなど\n"
        "   - 対象: 子ども、先生、保護者、おもちゃ、絵本など\n"
        "   - 感情・表情: 笑顔、泣いている、集中、楽しそうなど\n"
        "   例: [\"室内\", \"ブロック\", \"笑顔\", \"男の子\", \"遊び\"]\n\n"
        
        "3. scores: 各スコアは0.0から1.0の間の浮動小数点数で評価してください。\n"
        "   - cuteness: 写真の「可愛さ」や「愛らしさ」の度合い（0.0=普通、1.0=非常に可愛い）\n"
        "   - interesting: 写真の「興味深さ」や「印象的さ」の度合い（0.0=普通、1.0=非常に興味深い）\n"
        "   - representative: その日の活動全体を代表しているかどうか（0.0=特殊な場面、1.0=典型的な場面）\n\n"
        
        "4. has_child: 子どもが1人以上写っている場合はtrue、写っていない場合はfalse。\n\n"
        
        "5. num_children: 写っている子どもの人数を推定してください。0以上の整数で。\n\n"
        
        "6. main_subject: 主な被写体を短く表してください。\n"
        "   例: \"男の子\", \"女の子\", \"先生\", \"保護者\", \"複数の子ども\", \"おもちゃ\"など\n\n"
        
        "7. bbox: 主な子ども1人の位置をバウンディングボックスで指定してください。\n"
        "   画像全体を幅1.0、高さ1.0としたときの正規化座標で [x_min, y_min, x_max, y_max] の形式。\n"
        "   左上が原点(0,0)、右下が(1,1)です。\n"
        "   子どもが写っていない場合は [0.0, 0.0, 0.0, 0.0] を返してください。\n"
        "   例: 画像の中央付近にいる場合 [0.35, 0.40, 0.65, 0.80]\n\n"
        
        "【重要な注意事項】\n"
        "- 出力は必ずJSONオブジェクト1つだけにしてください。\n"
        "- JSONの前後に説明文、コメント、マークダウン記号（```jsonなど）は一切付けないでください。\n"
        "- すべてのキー（caption, tags, scores, has_child, num_children, main_subject, bbox）を含めてください。\n"
        "- JSONの構文エラーがないよう、ダブルクォート、カンマ、括弧を正確に使用してください。\n"
        "- 数値は浮動小数点数または整数として正しく記述してください。\n"
    )


def build_diary_prompt(frame_analyses: List[FrameAnalysis], max_chars: int, language: str = "ja") -> str:
    """
    フレームの caption 群から 1日のミニ日記を書かせるプロンプト。
    Llama4向けに最適化された高品質なプロンプト。
    """
    lines = [f"- {fa.caption}" for fa in frame_analyses]
    joined = "\n".join(lines)
    
    if language == "ja":
        instr = (
            "【タスク】\n"
            "上記の箇条書きは、保育園で撮影されたある1日の映像を数秒ごとに要約した文です。\n"
            f"これらをもとに、{max_chars}文字以内の日本語で「今日の様子」を温かく、読みやすい日記形式でまとめてください。\n\n"
            
            "【書き方のガイドライン】\n"
            "1. 時系列の流れを意識してください。朝から昼、午後へと時間が進むように自然な順序で記述してください。\n"
            "2. 細かすぎる描写は省き、印象的で重要な場面を中心にまとめてください。\n"
            "3. 温かく、親しみやすいトーンで書いてください。保護者が読んでほっこりできるような文章にしてください。\n"
            "4. 具体的な活動や様子を簡潔に伝えてください。\n"
            "5. 段落分けは不要です。1つの連続した文章として書いてください。\n"
            f"6. 文字数制限（{max_chars}文字以内）を厳守してください。\n\n"
            
            "【出力形式】\n"
            "- マークダウン記号や装飾は不要です。\n"
            "- 純粋な日本語のテキストのみを出力してください。\n"
            "- タイトルや見出しは付けないでください。\n"
            "- 日記本文のみを出力してください。\n\n"
            
            "【例】\n"
            "今日は朝から元気いっぱいでした。室内でブロック遊びを楽しみ、お友達と協力して大きな塔を作りました。"
            "お昼ご飯の時間には、みんなで「いただきます」をして、おいしそうに食べていました。"
            "午後は園庭で体を動かし、滑り台やブランコで遊びました。笑顔がたくさん見られて、とても充実した1日でした。"
        )
    else:
        instr = (
            "【Task】\n"
            "The bullet points above are captions describing moments from a single day at a daycare center.\n"
            f"Write a warm, short diary-style summary in {language} within {max_chars} characters.\n\n"
            
            "【Guidelines】\n"
            "1. Follow a natural chronological flow from morning to afternoon.\n"
            "2. Focus on memorable and important moments, omitting overly detailed descriptions.\n"
            "3. Write in a warm, friendly tone that parents would find heartwarming.\n"
            "4. Convey specific activities and situations concisely.\n"
            "5. Write as a single continuous text without paragraphs or headings.\n"
            f"6. Strictly adhere to the character limit ({max_chars} characters).\n\n"
            
            "【Output Format】\n"
            "- Output only plain text, no markdown or decorations.\n"
            "- No title or headings.\n"
            "- Output only the diary body text."
        )

    return joined + "\n\n" + instr
