"""
JSON Lines (JSONL) ファイルの読み書きヘルパ。
schemas で定義した dataclass とも連携できるようにする。
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, List, Type, TypeVar

import importlib
import schemas
importlib.reload(schemas)  # Colab用

T = TypeVar("T")


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    """
    records: dict or dataclass の iterable
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            if is_dataclass(r):
                obj = asdict(r)
            else:
                obj = r
            json_line = json.dumps(obj, ensure_ascii=False)
            f.write(json_line + "\n")


def read_jsonl_as_dicts(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def read_jsonl_as_dataclasses(path: Path, cls: Type[T]) -> List[T]:
    """
    指定した dataclass 型として JSONL を読み込む。
    """
    dicts = read_jsonl_as_dicts(path)
    return [cls(**d) for d in dicts]
