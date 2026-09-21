from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parents[1] / "ai" / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    purpose: str
    text: str


class PromptRegistry:
    def __init__(self, prompt_dir: Path = PROMPT_DIR) -> None:
        self.prompt_dir = prompt_dir

    def get(self, name: str, version: str) -> Prompt:
        path = self.prompt_dir / f"{name}_{version}.txt"
        text = path.read_text(encoding="utf-8")
        metadata = self._metadata(text)
        return Prompt(
            name=metadata.get("name", name),
            version=metadata.get("version", version),
            purpose=metadata.get("purpose", "airline assistant prompt"),
            text=text,
        )

    def _metadata(self, text: str) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for line in text.splitlines():
            if not line.startswith("# "):
                continue
            key, _, value = line[2:].partition(":")
            if key and value:
                metadata[key.strip().lower()] = value.strip()
        return metadata
