# memory/manager.py

import os
import json
from datetime import datetime
from typing import List, Dict

class MemoryManager:
    def __init__(
        self,
        episodic_path: str = "memory/episodic.json",
        semantic_path: str = "memory/semantic.json",
    ):
        """
        Manages episodic and semantic memory storage.
        """
        self.episodic_path = episodic_path
        self.semantic_path = semantic_path

        episodic_dir = os.path.dirname(self.episodic_path)
        semantic_dir = os.path.dirname(self.semantic_path)
        if episodic_dir:
            os.makedirs(episodic_dir, exist_ok=True)
        if semantic_dir:
            os.makedirs(semantic_dir, exist_ok=True)

        for path in (self.episodic_path, self.semantic_path):
            if not os.path.exists(path):
                with open(path, "w") as f:
                    json.dump([], f)

    def add_episodic(self, thought: str, salience: float = 0.5) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "thought": thought,
            "salience": salience
        }
        data = self._load(self.episodic_path)
        data.append(entry)
        self._save(self.episodic_path, data)

    def get_recent_memories(self, n: int = 5) -> List[str]:
        data = self._load(self.episodic_path)
        data_sorted = sorted(data, key=lambda e: e["timestamp"], reverse=True)
        return [e["thought"] for e in data_sorted[:n]]

    def consolidate_semantic(self) -> None:
        episodic = self._load(self.episodic_path)
        high = [e for e in episodic if e.get("salience", 0) >= 0.7]
        semantic = [{"timestamp": e["timestamp"], "summary": e["thought"]} for e in high]
        self._save(self.semantic_path, semantic)

    def _load(self, path: str) -> List[Dict]:
        with open(path, "r") as f:
            return json.load(f)

    def _save(self, path: str, data: List[Dict]) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)