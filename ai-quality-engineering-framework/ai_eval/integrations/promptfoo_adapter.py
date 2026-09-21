from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ai_eval.config import AIEvalSettings
from ai_eval.models import EvaluationResult, FrameworkExecutionReport


PROMPTFOO_CONFIG_PATH = Path("promptfoo") / "promptfooconfig.yaml"
PROMPTFOO_RESULTS_PATH = Path("reports") / "ai_eval" / "promptfoo-results.json"
PROMPTFOO_STATE_PATH = Path("promptfoo") / "generated" / "state"


class PromptfooAdapter:
    framework = "promptfoo"

    def __init__(self, *, settings: AIEvalSettings) -> None:
        self.settings = settings

    def validate_config(self, path: Path = PROMPTFOO_CONFIG_PATH) -> bool:
        return path.exists() and path.read_text(encoding="utf-8").strip() != ""

    def evaluate(self) -> FrameworkExecutionReport:
        installed, version = promptfoo_available()
        config_valid = self.validate_config()
        metadata_reason = "" if config_valid else "Promptfoo config is missing or empty."
        if not installed:
            return FrameworkExecutionReport(
                framework=self.framework,
                installed=False,
                enabled=self.settings.promptfoo_enabled,
                provider="deterministic-local",
                status="unavailable",
                version=version,
                reason=(
                    metadata_reason
                    or "Promptfoo CLI is not installed or not available on PATH."
                ),
            )
        if not self.settings.promptfoo_enabled:
            return FrameworkExecutionReport(
                framework=self.framework,
                installed=True,
                enabled=False,
                provider="deterministic-local",
                status="skipped",
                version=version,
                reason="Promptfoo is available but AI_EVAL_PROMPTFOO_ENABLED is false.",
            )

        if PROMPTFOO_RESULTS_PATH.exists():
            try:
                results, summary = _load_promptfoo_results(PROMPTFOO_RESULTS_PATH)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return FrameworkExecutionReport(
                    framework=self.framework,
                    installed=True,
                    enabled=True,
                    provider="deterministic-local",
                    status="failed",
                    version=version,
                    reason=f"Promptfoo result artifact could not be read: {exc}",
                )
            return FrameworkExecutionReport(
                framework=self.framework,
                installed=True,
                enabled=True,
                provider="deterministic-local",
                status="executed",
                version=version,
                results=results,
                reason=(
                    f"Promptfoo executed {summary['total']} cases: "
                    f"{summary['passed']} passed, {summary['failed']} failed, "
                    f"{summary['errors']} errors. Artifact: {PROMPTFOO_RESULTS_PATH}"
                ),
                metadata=summary,
            )

        return FrameworkExecutionReport(
            framework=self.framework,
            installed=True,
            enabled=True,
            provider="deterministic-local",
            status="not_executed",
            version=version,
            reason=(
                "Promptfoo is enabled but no result artifact exists. Run the "
                "project-local Promptfoo evaluation first."
            ),
        )


def promptfoo_available() -> tuple[bool, str | None]:
    local_executable = Path("promptfoo") / "node_modules" / ".bin" / (
        "promptfoo.cmd" if os.name == "nt" else "promptfoo"
    )
    executable = (
        str(local_executable.resolve())
        if local_executable.exists()
        else shutil.which("promptfoo") or shutil.which("promptfoo.cmd")
    )
    if executable is None:
        return False, None
    try:
        env = os.environ.copy()
        state_path = PROMPTFOO_STATE_PATH.resolve()
        state_path.mkdir(parents=True, exist_ok=True)
        env.setdefault("PROMPTFOO_CONFIG_DIR", str(state_path))
        env.setdefault("PROMPTFOO_DISABLE_WAL_MODE", "true")
        env.setdefault("PROMPTFOO_DISABLE_TELEMETRY", "true")
        completed = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, None
    version = completed.stdout.strip() or completed.stderr.strip() or None
    return completed.returncode == 0, version


def _load_promptfoo_results(
    path: Path,
) -> tuple[list[EvaluationResult], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_results = payload["results"]["results"]
    results: list[EvaluationResult] = []
    passed = failed = errors = 0
    for item in raw_results:
        case_id = item.get("vars", {}).get("case_id") or item.get("id", "unknown")
        grading = item.get("gradingResult") or {}
        error = item.get("error") or item.get("response", {}).get("error")
        if error:
            errors += 1
            execution_status = "failed"
            did_pass = False
            reason = str(error)
        else:
            did_pass = bool(item.get("success", grading.get("pass", False)))
            passed += int(did_pass)
            failed += int(not did_pass)
            execution_status = "executed"
            reason = str(grading.get("reason", ""))
        category = item.get("vars", {}).get("category")
        metric = {
            "pii": "pii_protection",
            "prompt_injection": "prompt_injection_resistance",
            "hallucination": "hallucination_protection",
            "unsupported_request": "safe_refusal",
            "safety": "safe_refusal",
        }.get(category, "deterministic_assertions")
        results.append(
            EvaluationResult(
                framework="promptfoo",
                metric=metric,
                case_id=str(case_id),
                score=item.get("score", grading.get("score")),
                threshold=1.0,
                passed=did_pass,
                reason=reason,
                provider="deterministic-local",
                execution_status=execution_status,
                metadata={
                    "category": category,
                    "result_id": item.get("id"),
                },
            )
        )
    return results, {
        "total": len(raw_results),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "eval_id": payload.get("evalId"),
        "provider": "deterministic-local-airline-provider",
        "evaluation_type": "normal_with_adversarial_cases",
        "red_team_executed": False,
        "artifact": str(path),
    }
