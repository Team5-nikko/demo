"""
API キーをどこから読むかをまとめるヘルパー。

優先順位:
1. 環境変数（Cloud Run / ローカル実行）
2. Colab の userdata（開発用）
3. Streamlit の secrets（ローカル開発用、ある場合のみ）

最終的に os.environ にセットされていればOK。
"""

import os


def init_gemini_api_key() -> None:
    # すでに設定されていれば何もしない（Cloud Run など）
    if os.environ.get("GEMINI_API_KEY"):
        return

    # Colab の userdata から読む（開発用）
    try:
        from google.colab import userdata  # type: ignore
        key = userdata.get("GEMINI_API_KEY")
        if key:
            os.environ["GEMINI_API_KEY"] = key
            return
    except Exception:
        pass

    # Streamlit の secrets から読む（ローカル実行用）
    try:
        import streamlit as st  # type: ignore
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
            os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
            return
    except Exception:
        pass

    # ここまで来ても無ければエラー
    raise RuntimeError(
        "GEMINI_API_KEY がどこにも設定されていません。\n"
        "- Cloud Run: 環境変数に Secret をマウント\n"
        "- Colab: userdata に GEMINI_API_KEY を保存\n"
        "- ローカル: .streamlit/secrets.toml に GEMINI_API_KEY を設定"
    )


def init_sambanova_api_key() -> None:
    """
    SambaNova API キーを初期化する。
    優先順位は init_gemini_api_key と同じ。
    """
    # すでに設定されていれば何もしない（Cloud Run など）
    if os.environ.get("SAMBANOVA_API_KEY"):
        return

    # Colab の userdata から読む（開発用）
    try:
        from google.colab import userdata  # type: ignore
        key = userdata.get("SAMBANOVA_API_KEY")
        if key:
            os.environ["SAMBANOVA_API_KEY"] = key
            return
    except Exception:
        pass

    # Streamlit の secrets から読む（ローカル実行用）
    try:
        import streamlit as st  # type: ignore
        if "SAMBANOVA_API_KEY" in st.secrets and st.secrets["SAMBANOVA_API_KEY"]:
            os.environ["SAMBANOVA_API_KEY"] = st.secrets["SAMBANOVA_API_KEY"]
            return
    except Exception:
        pass

    # ここまで来ても無ければエラー
    raise RuntimeError(
        "SAMBANOVA_API_KEY がどこにも設定されていません。\n"
        "- Cloud Run: 環境変数に Secret をマウント\n"
        "- Colab: userdata に SAMBANOVA_API_KEY を保存\n"
        "- ローカル: .streamlit/secrets.toml に SAMBANOVA_API_KEY を設定"
    )
