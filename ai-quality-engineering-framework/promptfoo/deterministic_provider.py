from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_quality.models import GoldenCase
from ai_quality.prompt_registry import PromptRegistry
from ai_quality.providers import DeterministicLLMProvider


def main() -> int:
    raw = sys.stdin.read().strip()
    payload = json.loads(raw) if raw else _payload_from_args(sys.argv[1:])
    vars_payload = payload.get("vars", {})
    prompt_payload = payload.get("prompt", {})
    prompt = (
        vars_payload.get("user_input")
        or payload.get("prompt")
        or prompt_payload.get("raw")
        or prompt_payload.get("template")
        or ""
    )
    case_payload = vars_payload.get("golden_case")

    case = GoldenCase.model_validate_json(case_payload) if case_payload else None
    provider = DeterministicLLMProvider()
    registry_prompt = PromptRegistry().get("airline_assistant", "v1")
    response = provider.generate_structured(
        prompt,
        prompt=registry_prompt,
        case=case,
        context=case.context if case else None,
    )
    sys.stdout.write(response.response)
    return 0


def _payload_from_args(args: list[str]) -> dict:
    for arg in reversed(args):
        try:
            payload = json.loads(arg)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and ("vars" in payload or "prompt" in payload):
            return payload
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
