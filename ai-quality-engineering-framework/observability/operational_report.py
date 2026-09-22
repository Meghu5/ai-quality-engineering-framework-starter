from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from ai_eval.models import Phase10Report
from observability.analysis import OperationalAnalysisReport, analyze_operational_evidence
from observability.ci import CI_REPORT_PATH, CiCorrelationReport, TRACE_REPORT_PATH
from observability.models import TraceEnvelope


logger = logging.getLogger(__name__)
PHASE10_REPORT_PATH = Path("reports") / "ai_eval" / "phase10_report.json"
OPERATIONAL_REPORT_PATH = (
    Path("reports") / "observability" / "operational-analysis.json"
)
MAX_ARTIFACT_BYTES = 25 * 1024 * 1024


def build_operational_report_from_files(
    *,
    trace_path: Path = TRACE_REPORT_PATH,
    ci_path: Path = CI_REPORT_PATH,
    phase10_path: Path = PHASE10_REPORT_PATH,
) -> OperationalAnalysisReport:
    traces = _load_optional_models(trace_path, TraceEnvelope)
    ci_report = _load_optional_model(ci_path, CiCorrelationReport)
    phase10_report = _load_optional_model(phase10_path, Phase10Report)
    return analyze_operational_evidence(
        traces, ci_report=ci_report, phase10_report=phase10_report
    )


def write_operational_report(
    path: Path = OPERATIONAL_REPORT_PATH,
    *,
    trace_path: Path = TRACE_REPORT_PATH,
    ci_path: Path = CI_REPORT_PATH,
    phase10_path: Path = PHASE10_REPORT_PATH,
) -> OperationalAnalysisReport:
    report = build_operational_report_from_files(
        trace_path=trace_path, ci_path=ci_path, phase10_path=phase10_path
    )
    _write_json_atomic(path, report.model_dump(mode="json"))
    return report


def main() -> int:
    try:
        write_operational_report()
    except Exception as exc:
        logger.error(
            "Unable to generate operational observability report (%s)",
            type(exc).__name__,
        )
    return 0


def _load_optional_models(path: Path, model_type: type) -> list:
    if not path.exists():
        return []
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path.name} must contain a JSON array")
    return [model_type.model_validate(item) for item in payload]


def _load_optional_model(path: Path, model_type: type) -> Any | None:
    if not path.exists():
        return None
    return model_type.model_validate(_read_json(path))


def _read_json(path: Path) -> Any:
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"{path.name} exceeds the evidence size limit")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
