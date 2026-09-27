"""配置记忆模块：以 JSON 持久化用户参数与路径，重启后自动恢复。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AppMemory:
    """应用配置记忆，读写 JSON 文件。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        """读取配置；文件不存在或损坏时返回空字典。"""
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}
        return self._data

    def save(self) -> None:
        """将当前配置写回文件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def add_recent(self, key: str, value: str, maxlen: int = 5) -> None:
        """把 value 记入 key 对应的最近使用列表（去重、最新在前，最多 maxlen 条）。"""
        recent = self._data.get(key, [])
        if not isinstance(recent, list):
            recent = []
        recent = [v for v in recent if v != value]
        recent.insert(0, value)
        self._data[key] = recent[:maxlen]

    def get_recent(self, key: str) -> list[str]:
        recent = self._data.get(key, [])
        return list(recent) if isinstance(recent, list) else []
