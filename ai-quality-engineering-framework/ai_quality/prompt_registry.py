from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from weakref import WeakKeyDictionary


PROMPT_DIR = Path(__file__).resolve().parents[1] / "ai" / "prompts"
UNREGISTERED_PROMPT_ID = "unregistered"
UNREGISTERED_PROMPT_VERSION = "0"
_REGISTERED_PROMPT_PATHS = MappingProxyType(
    {
        ("airline_assistant", "v1"): PROMPT_DIR / "airline_assistant_v1.txt",
        ("airline_assistant", "v2"): PROMPT_DIR / "airline_assistant_v2.txt",
    }
)
REGISTERED_PROMPT_IDENTITIES = frozenset(_REGISTERED_PROMPT_PATHS)


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    purpose: str
    text: str


class RegisteredPrompt:
    __slots__ = ("_prompt", "_capability", "__weakref__")

    def __new__(cls, *args, **kwargs):
        raise TypeError("registered prompts must be resolved by PromptRegistry")

    @property
    def prompt(self) -> Prompt:
        return self._prompt

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("registered prompts are immutable")

    def __repr__(self) -> str:
        return (
            "RegisteredPrompt("
            f"name={self._prompt.name!r}, version={self._prompt.version!r})"
        )


_ISSUED_PROMPTS: WeakKeyDictionary[RegisteredPrompt, object] = WeakKeyDictionary()


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

    def resolve(self, name: str, version: str) -> RegisteredPrompt:
        path = _REGISTERED_PROMPT_PATHS.get((name, version))
        if path is None:
            raise KeyError("prompt registration not found")
        text = path.read_text(encoding="utf-8")
        metadata = self._metadata(text)
        if metadata.get("name") != name or metadata.get("version") != version:
            raise ValueError("registered prompt metadata is inconsistent")
        prompt = Prompt(
            name=name,
            version=version,
            purpose=metadata.get("purpose", "airline assistant prompt"),
            text=text,
        )
        return _issue_registered_prompt(prompt)

    def _metadata(self, text: str) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for line in text.splitlines():
            if not line.startswith("# "):
                continue
            key, _, value = line[2:].partition(":")
            if key and value:
                metadata[key.strip().lower()] = value.strip()
        return metadata


def registered_prompt_identity(
    prompt: Prompt | RegisteredPrompt,
) -> tuple[str, str]:
    if isinstance(prompt, RegisteredPrompt) and _is_issued(prompt):
        return prompt.prompt.name, prompt.prompt.version
    return UNREGISTERED_PROMPT_ID, UNREGISTERED_PROMPT_VERSION


def prompt_for_execution(prompt: Prompt | RegisteredPrompt) -> Prompt:
    if isinstance(prompt, RegisteredPrompt):
        if not _is_issued(prompt):
            raise ValueError("registered prompt capability is invalid")
        return prompt.prompt
    if isinstance(prompt, Prompt):
        return prompt
    raise TypeError("prompt must be Prompt or RegisteredPrompt")


def _issue_registered_prompt(prompt: Prompt) -> RegisteredPrompt:
    handle = object.__new__(RegisteredPrompt)
    object.__setattr__(handle, "_prompt", prompt)
    capability = object()
    object.__setattr__(handle, "_capability", capability)
    _ISSUED_PROMPTS[handle] = capability
    return handle


def _is_issued(prompt: RegisteredPrompt) -> bool:
    return _ISSUED_PROMPTS.get(prompt) is prompt._capability
