"""
シンプルな Streamlit アプリ

- 動画ファイルをアップロード
- video_id を自動 or 手動で決定
- バックエンドのパイプラインを一気に実行
  - 動画 → フレーム抽出 → 前処理 → マニフェスト
  - Vision 解析（Gemini / ダミー） → ベストショット選定 → 日記生成
- 結果としてベストショット画像と日記テキストを表示
- さらに各ステップの成果物を「デバッグビュー」としてクリック展開で確認可能

※ バックエンド部分は Colab 上で既に作成済みの各 *.py を利用する。
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

import config_loader
import paths
import video_loader
import frame_extractor
import frame_preprocessor
import manifest_builder
import vision_captioner
import bestshot_scorer
import diary_generator
import inspection
import jsonl_io

# # Colab / Streamlit の secrets から GEMINI_API_KEY を拾って env に入れる（あれば）
# if "GEMINI_API_KEY" in st.secrets:
#     os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# Colab で他の py を編集したときも、毎回最新を読むために reload しておく
importlib.reload(config_loader)
importlib.reload(paths)
importlib.reload(video_loader)
importlib.reload(frame_extractor)
importlib.reload(frame_preprocessor)
importlib.reload(manifest_builder)
importlib.reload(vision_captioner)
importlib.reload(bestshot_scorer)
importlib.reload(diary_generator)
importlib.reload(inspection)
importlib.reload(jsonl_io)

from config_loader import SETTINGS
from video_loader import save_video, generate_video_id
from frame_extractor import extract_frames
from frame_preprocessor import preprocess_frames
from manifest_builder import build_manifest
from vision_captioner import run_captioning
from bestshot_scorer import select_bestshots
from diary_generator import generate_diary
from inspection import show_sample_frames
from paths import (
    get_bestshot_meta_path,
    get_diary_path,
    get_manifest_path,
    get_analysis_path,
    get_frames_dir,
    get_raw_video_dir,
)
from jsonl_io import read_jsonl_as_dicts


def run_full_pipeline(video_file, custom_video_id: Optional[str] = None) -> str:
    """
    アップロード動画を受け取り、パイプラインを最後まで実行するヘルパ関数。
    戻り値: 実際に使用した video_id
    """
    # video_id を決定
    if custom_video_id and custom_video_id.strip():
        video_id = custom_video_id.strip()
    else:
        video_id = generate_video_id(prefix="ui")

    suffix = Path(video_file.name).suffix or ".mp4"

    # 1. 動画保存
    st.write("### 1. 動画を保存しています …")
    video_id = save_video(video_file, video_id=video_id, suffix=suffix)
    st.write(f"- video_id: `{video_id}`")

    # 2. フレーム抽出
    st.write("### 2. フレームを抽出しています …")
    with st.spinner("フレーム抽出中…"):
        frames_meta = extract_frames(video_id)
    st.write(f"- 抽出フレーム枚数: {len(frames_meta)}")

    # 3. フレーム前処理
    st.write("### 3. フレームを前処理しています …")
    with st.spinner("前処理中…"):
        frames_meta = preprocess_frames(frames_meta)

    # 4. マニフェスト作成
    st.write("### 4. マニフェストを作成しています …")
    build_manifest(video_id, frames_meta)

    # 5. 画像解析（Gemini または ダミー）
    st.write("### 5. 画像解析を実行しています …")
    with st.spinner("画像解析中…"):
        _ = run_captioning(video_id)

    # 6. ベストショット選定
    st.write("### 6. ベストショットを選定しています …")
    with st.spinner("ベストショット選定中…"):
        _ = select_bestshots(video_id)

    # 7. 日記生成
    st.write("### 7. 日記テキストを生成しています …")
    with st.spinner("日記生成中…"):
        _ = generate_diary(video_id)

    st.success("パイプラインが完了しました。")
    return video_id


def main():
    st.set_page_config(page_title="子ども見守りダイジェスト", layout="wide")
    st.title("子ども見守りダイジェスト")
    st.caption(
        "動画をアップロードして、ベストショットとミニ日記を自動生成します。"
        "※ LLM のバックエンドは config/models.yaml の設定（Gemini / dummy / local）に従います。"
    )

    st.sidebar.header("設定")
    st.sidebar.write(f"データ保存先: `{SETTINGS.data_root}`")
    st.sidebar.write(f"フレーム間隔: {SETTINGS.frame_interval_sec} 秒")
    st.sidebar.write(f"ベストショット最大枚数: {SETTINGS.max_bestshots}")

    st.sidebar.markdown("---")
    st.sidebar.write("動画の基本情報を入力してください。")

    custom_video_id = st.sidebar.text_input(
        "video_id（任意。空なら自動生成）",
        value="",
        help="指定すると outputs 以下のフォルダ名などに利用されます。",
    )

    st.markdown("## 1. 動画をアップロード")

    video_file = st.file_uploader(
        "動画ファイルを選択してください（mp4 / mov / avi / mkv）",
        type=["mp4", "mov", "avi", "mkv"],
    )

    run_button = st.button("この動画を解析する", type="primary", disabled=(video_file is None))

    if video_file is None:
        st.info("左のボタンから動画ファイルをアップロードしてください。")
        return

    if run_button:
        try:
            # パイプライン実行
            video_id = run_full_pipeline(video_file, custom_video_id=custom_video_id)

            st.markdown("---")
            st.markdown(f"## 2. 結果（video_id: `{video_id}`）")

            # ベストショット表示
            st.subheader("ベストショット")
            meta_path = get_bestshot_meta_path(video_id)
            metas = []
            if meta_path.exists():
                metas = json.loads(meta_path.read_text(encoding="utf-8"))
                cols = st.columns(3)
                for i, m in enumerate(metas):
                    col = cols[i % len(cols)]
                    with col:
                        st.image(m["frame_path"], caption=f"#{m['rank']} - {m['caption']}")
            else:
                st.write("ベストショット情報が見つかりませんでした。")

            # 日記表示
            st.subheader("今日のミニ日記")
            diary_path = get_diary_path(video_id)
            if diary_path.exists():
                diary_text = diary_path.read_text(encoding="utf-8")
                st.markdown(diary_text)
            else:
                st.write("日記ファイルが見つかりませんでした。")

            # --- ここからデバッグビュー ---
            st.markdown("---")
            st.markdown("## 3. デバッグビュー（中間生成物を確認）")

            # 3-1. 保存された動画ファイル
            with st.expander("① 保存された動画ファイルを確認する"):
                raw_dir = get_raw_video_dir()
                candidates = list(raw_dir.glob(f"{video_id}*"))
                if candidates:
                    video_path = candidates[0]
                    st.write(f"パス: `{video_path}`")
                    st.video(str(video_path))
                else:
                    st.write("対応する動画ファイルが見つかりませんでした。")

            # 3-2. 抽出フレーム
            with st.expander("② 抽出されたフレームを確認する"):
                frames_dir = get_frames_dir(video_id)
                frame_files = sorted(frames_dir.glob("*.png"))
                st.write(f"フレーム枚数: {len(frame_files)}")
                if frame_files:
                    # 最初の数枚だけ表示
                    max_show = min(12, len(frame_files))
                    cols = st.columns(4)
                    for i, p in enumerate(frame_files[:max_show]):
                        img = Image.open(p)
                        col = cols[i % len(cols)]
                        with col:
                            st.image(img, caption=p.name)
                else:
                    st.write("フレーム画像が見つかりませんでした。")

            # 3-3. マニフェスト JSONL
            with st.expander("③ マニフェスト（frames_manifest.jsonl）の中身を見る"):
                manifest_path = get_manifest_path(video_id)
                if manifest_path.exists():
                    records = read_jsonl_as_dicts(manifest_path)
                    st.write(f"レコード数: {len(records)}")
                    if records:
                        st.json(records[:5])  # 先頭5件だけ
                else:
                    st.write("マニフェストファイルが見つかりませんでした。")

            # 3-4. 画像解析結果 JSONL
            with st.expander("④ 画像解析結果（analysis.jsonl）の中身を見る"):
                analysis_path = get_analysis_path(video_id)
                if analysis_path.exists():
                    records = read_jsonl_as_dicts(analysis_path)
                    st.write(f"レコード数: {len(records)}")
                    if records:
                        st.json(records[:5])  # 先頭5件だけ
                else:
                    st.write("解析結果ファイルが見つかりませんでした。")

            # 3-5. ベストショットメタ情報
            with st.expander("⑤ ベストショットのメタ情報を見る"):
                if metas:
                    st.json(metas)
                else:
                    st.write("ベストショットメタ情報が存在しません。")

            # 3-6. 元フレームのランダムサンプル（既存の inspection 利用）
            with st.expander("⑥ 元フレームをランダム表示（inspection.show_sample_frames）"):
                st.write("下にランダムで数枚のフレームを表示します。")
                show_sample_frames(video_id, n=4)

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
