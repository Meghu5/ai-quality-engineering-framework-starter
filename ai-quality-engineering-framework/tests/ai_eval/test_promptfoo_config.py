from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ai_eval.config import AIEvalSettings
from ai_eval.integrations.promptfoo_adapter import PromptfooAdapter, _load_promptfoo_results
from promptfoo.build_promptfoo_config import build_promptfoo_tests, write_promptfoo_tests


pytestmark = pytest.mark.ai_eval


def test_promptfoo_config_references_deterministic_provider_and_generated_tests():
    config = yaml.safe_load(Path("promptfoo/promptfooconfig.yaml").read_text(encoding="utf-8"))

    assert config["providers"][0]["id"].startswith("exec:python")
    assert config["tests"] == "generated/airline_regression.yaml"


def test_promptfoo_generated_tests_come_from_existing_ai_dataset(tmp_path):
    output = tmp_path / "airline_regression.yaml"
    tests = write_promptfoo_tests(output)
    generated = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert len(tests) >= 25
    assert len(generated) >= 25
    assert generated[0]["vars"]["case_id"].startswith("ai-")
    assert "golden_case" in generated[0]["vars"]


def test_promptfoo_adapter_reports_unexecuted_without_fabricated_scores(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "ai_eval.integrations.promptfoo_adapter.PROMPTFOO_RESULTS_PATH",
        tmp_path / "missing-results.json",
    )
    adapter = PromptfooAdapter(settings=AIEvalSettings(False, False, True, "none"))
    report = adapter.evaluate()

    assert report.framework == "promptfoo"
    assert report.status in {"unavailable", "not_executed"}
    assert report.results == []


def test_promptfoo_build_has_security_cases():
    tests = build_promptfoo_tests()
    security_descriptions = [
        test["description"]
        for test in tests
        if test["vars"]["category"] in {"prompt_injection", "pii", "hallucination", "unsupported_request"}
    ]

    assert any("prompt-injection" in description for description in security_descriptions)
    assert any("pii" in description for description in security_descriptions)


def test_promptfoo_pii_case_requires_safe_handling_and_rejects_synthetic_echo():
    pii_case = next(
        test for test in build_promptfoo_tests() if test["vars"]["category"] == "pii"
    )
    assertions = pii_case["assert"]
    forbidden_values = {
        assertion["value"]
        for assertion in assertions
        if assertion["type"] == "not-icontains"
    }

    assert any(
        assertion["type"] == "icontains-any" and "airline" in assertion["value"]
        for assertion in assertions
    )
    assert forbidden_values == {
        "test@example.com",
        "TEST-CARD-4111111111111111",
    }

    unsafe_echo = (
        "I can help with airline travel for test@example.com using "
        "TEST-CARD-4111111111111111."
    )
    assert any(value.casefold() in unsafe_echo.casefold() for value in forbidden_values)


def test_promptfoo_result_artifact_is_normalized_without_fabricating_scores(tmp_path):
    artifact = tmp_path / "promptfoo-results.json"
    artifact.write_text(
        """{
          "results": {"results": [
            {
              "id": "result-1",
              "success": true,
              "score": 1,
              "gradingResult": {"pass": true, "score": 1, "reason": "All assertions passed"},
              "vars": {"case_id": "ai-019-prompt-injection", "category": "prompt_injection"}
            }
          ]}
        }""",
        encoding="utf-8",
    )

    results, summary = _load_promptfoo_results(artifact)

    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["errors"] == 0
    assert summary["red_team_executed"] is False
    assert results[0].execution_status == "executed"
    assert results[0].provider == "deterministic-local"
    assert results[0].metadata["category"] == "prompt_injection"
