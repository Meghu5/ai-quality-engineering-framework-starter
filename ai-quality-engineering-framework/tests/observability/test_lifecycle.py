from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import observability.lifecycle as lifecycle
from ai_eval.models import EvaluationResult, FrameworkExecutionReport, Phase10Report
from observability.analysis import analyze_operational_evidence
from observability.ci import CiCorrelationReport, CiTestEvidence
from observability.lifecycle import (
    ArtifactType,
    RunIdSource,
    begin_run,
    build_manifest,
    create_bundle_from_existing_reports,
    finalize_run,
    plan_retention,
    prune_runs,
    stage_artifacts,
    validate_artifact,
    verify_bundle,
)
from observability.models import SpanEvidence, TraceEnvelope, TraceStatus
from observability.readiness import analyze_observability_readiness


pytestmark = pytest.mark.observability


def _span(index: int = 1, **overrides) -> SpanEvidence:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    values = {
        "trace_id": f"{index:032x}",
        "span_id": f"{index:016x}",
        "correlation_id": f"correlation-{index}",
        "operation_name": "observability.lifecycle",
        "operation_type": "internal",
        "started_at": started,
        "ended_at": started + timedelta(milliseconds=index),
        "duration_ms": float(index),
        "status": TraceStatus.OK,
        "attributes": {"case_id": f"case-{index}"},
        "service_name": "quality-engine",
        "environment": "test",
    }
    values.update(overrides)
    return SpanEvidence(**values)


def _trace(index: int = 1) -> TraceEnvelope:
    span = _span(index)
    return TraceEnvelope(
        trace_id=span.trace_id,
        correlation_id=span.correlation_id,
        service_name=span.service_name,
        environment=span.environment,
        spans=[span],
    )


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    return path


def _write_trace_report(root: Path, index: int = 1) -> Path:
    trace = _trace(index)
    return _write_json(
        root / "observability" / "traces.json",
        [trace.model_dump(mode="json")],
    )


def _ci_report(trace_id: str) -> CiCorrelationReport:
    test = CiTestEvidence(
        test_id="a" * 64,
        node_id="tests/observability/test_lifecycle.py::test_bundle",
        outcome="passed",
        duration_ms=1.0,
        correlation_id="correlation-1",
        trace_ids=[trace_id],
        case_ids=["case-1"],
    )
    return CiCorrelationReport(
        tests=[test],
        failures=[],
        summary={
            "total_tests": 1,
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
            "failures_by_category": {},
            "tests_with_traces": 1,
            "tests_without_traces": 0,
        },
    )


def _phase10_report() -> Phase10Report:
    result = EvaluationResult(
        framework="baseline",
        metric="quality",
        case_id="case-1",
        score=1.0,
        threshold=0.8,
        passed=True,
        execution_status="executed",
    )
    framework = FrameworkExecutionReport(
        framework="baseline",
        installed=True,
        enabled=True,
        status="executed",
        results=[result],
    )
    return Phase10Report(baseline={}, frameworks=[framework], overall_passed=True)


def _write_complete_evidence(root: Path) -> dict[ArtifactType, Path]:
    trace = _trace(1)
    traces = [trace]
    ci_report = _ci_report(trace.trace_id)
    phase10_report = _phase10_report()
    analysis = analyze_operational_evidence(
        traces,
        ci_report=ci_report,
        phase10_report=phase10_report,
    )
    readiness = analyze_observability_readiness(
        analysis,
        traces,
        ci_report=ci_report,
    )
    return {
        ArtifactType.TRACES: _write_json(
            root / "observability" / "traces.json",
            [item.model_dump(mode="json") for item in traces],
        ),
        ArtifactType.CI_CORRELATION: _write_json(
            root / "ci" / "test-trace-correlation.json",
            ci_report.model_dump(mode="json"),
        ),
        ArtifactType.FAILURE_SUMMARY: _write_json(
            root / "ci" / "failure-summary.json",
            ci_report.summary,
        ),
        ArtifactType.PHASE10_REPORT: _write_json(
            root / "ai_eval" / "phase10_report.json",
            phase10_report.model_dump(mode="json"),
        ),
        ArtifactType.OPERATIONAL_ANALYSIS: _write_json(
            root / "observability" / "operational-analysis.json",
            analysis.model_dump(mode="json"),
        ),
        ArtifactType.READINESS: _write_json(
            root / "observability" / "readiness.json",
            readiness.model_dump(mode="json"),
        ),
    }


def _create_trace_bundle(tmp_path: Path, index: int = 1) -> Path:
    source_root = tmp_path / f"reports-{index}"
    lifecycle_root = tmp_path / "lifecycle"
    _write_trace_report(source_root, index)
    return create_bundle_from_existing_reports(
        source_root=source_root,
        lifecycle_root=lifecycle_root,
        required_types=[ArtifactType.TRACES],
    )


def test_bundle_lifecycle_stages_validates_and_finalizes_required_artifacts(tmp_path):
    source_root = tmp_path / "reports"
    lifecycle_root = tmp_path / "lifecycle"
    _write_trace_report(source_root)

    run = begin_run(lifecycle_root)
    artifacts = stage_artifacts(
        run,
        source_root=source_root,
        artifact_types=[ArtifactType.TRACES],
    )
    manifest = build_manifest(run, artifacts, required_types=[ArtifactType.TRACES])

    assert run.run_id == "local-000001"
    assert run.run_id_source == RunIdSource.LOCAL
    assert manifest.aggregate_record_count == 1
    assert manifest.artifacts[0].required is True

    final_path = finalize_run(run, artifacts, required_types=[ArtifactType.TRACES])
    verified = verify_bundle(final_path)

    assert final_path == lifecycle_root.resolve() / "runs" / run.run_id
    assert not run.staging_path.exists()
    assert verified.run_id == run.run_id
    assert verified.artifacts[0].artifact_type == ArtifactType.TRACES


def test_realistic_evidence_bundle_includes_and_verifies_every_step7_artifact(tmp_path):
    source_root = tmp_path / "reports"
    lifecycle_root = tmp_path / "lifecycle"
    evidence_paths = _write_complete_evidence(source_root)

    run = begin_run(lifecycle_root)
    artifacts = stage_artifacts(run, source_root=source_root)
    final_path = finalize_run(
        run,
        artifacts,
        required_types=ArtifactType,
    )
    manifest = verify_bundle(final_path)

    assert {item.artifact_type for item in manifest.artifacts} == set(ArtifactType)
    assert manifest.aggregate_byte_count == sum(
        path.stat().st_size for path in evidence_paths.values()
    )
    assert manifest.aggregate_record_count >= len(ArtifactType)
    assert all(item.required for item in manifest.artifacts)


def test_run_ids_are_stable_sanitized_and_do_not_expose_source_values(tmp_path):
    explicit = begin_run(
        tmp_path / "explicit",
        environment={"AI_OBSERVABILITY_RUN_ID": "customer@example.com:secret-token"},
    )
    ci = begin_run(
        tmp_path / "ci",
        environment={
            "GITHUB_RUN_ID": "123456",
            "GITHUB_RUN_ATTEMPT": "2",
            "GITHUB_JOB": "observability-tests",
        },
    )

    assert explicit.run_id_source == RunIdSource.EXPLICIT
    assert explicit.run_id.startswith("explicit-")
    assert "customer" not in explicit.run_id
    assert "secret" not in explicit.run_id
    assert ci.run_id_source == RunIdSource.CI
    assert ci.run_id.startswith("ci-")

    with pytest.raises(ValueError, match="decimal digits"):
        begin_run(tmp_path / "bad-ci", environment={"GITHUB_RUN_ID": "12/34"})


def test_ci_derived_identity_is_stable_per_attempt_and_job(tmp_path):
    environment = {
        "GITHUB_RUN_ID": "123456",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_JOB": "observability-tests",
    }
    first = begin_run(tmp_path / "first", environment=environment)
    second = begin_run(tmp_path / "second", environment=environment)
    third = begin_run(
        tmp_path / "third",
        environment={**environment, "GITHUB_RUN_ATTEMPT": "3"},
    )

    assert first.run_id == second.run_id
    assert first.run_id != third.run_id
    assert first.run_id_source == RunIdSource.CI
    assert re.fullmatch(r"ci-[0-9a-f]{24}", first.run_id)


def test_uuid_values_are_never_used_directly_as_run_identity(tmp_path):
    raw_uuid = "12345678-1234-5678-1234-567812345678"
    run = begin_run(
        tmp_path / "lifecycle",
        environment={"AI_OBSERVABILITY_RUN_ID": raw_uuid},
    )

    assert run.run_id_source == RunIdSource.EXPLICIT
    assert run.run_id != f"explicit-{raw_uuid}"
    assert raw_uuid not in run.run_id
    assert re.fullmatch(r"explicit-[0-9a-f]{24}", run.run_id)


def test_required_artifacts_and_schema_versions_are_enforced(tmp_path):
    source_root = tmp_path / "reports"
    lifecycle_root = tmp_path / "lifecycle"
    _write_trace_report(source_root)
    run = begin_run(lifecycle_root)
    artifacts = stage_artifacts(
        run,
        source_root=source_root,
        artifact_types=[ArtifactType.TRACES],
    )

    with pytest.raises(ValueError, match="required lifecycle artifacts are missing"):
        build_manifest(run, artifacts, required_types=[ArtifactType.READINESS])
    with pytest.raises(ValueError, match="unsupported schema version"):
        validate_artifact(
            source_root / "observability" / "traces.json",
            ArtifactType.TRACES,
            "9.9",
        )


@pytest.mark.parametrize("artifact_type", list(ArtifactType))
def test_schema_compatibility_for_supported_artifacts(tmp_path, artifact_type):
    evidence_paths = _write_complete_evidence(tmp_path / "reports")
    records = validate_artifact(
        evidence_paths[artifact_type],
        artifact_type,
        lifecycle.ARTIFACT_SPECS[artifact_type].schema_version,
    )

    assert records >= 0


def test_verify_bundle_rejects_manifest_tampering_and_path_escape(tmp_path):
    final_path = _create_trace_bundle(tmp_path)
    manifest_path = final_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["relative_path"] = "../traces.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="path is not allowlisted"):
        verify_bundle(final_path)


def test_checksum_corruption_is_rejected(tmp_path):
    final_path = _create_trace_bundle(tmp_path)
    artifact_path = final_path / "artifacts" / "observability" / "traces.json"
    original = artifact_path.read_text(encoding="utf-8")
    corrupted = original.replace(
        "00000000000000000000000000000001",
        "00000000000000000000000000000002",
        1,
    )
    assert len(corrupted.encode("utf-8")) == len(original.encode("utf-8"))
    artifact_path.write_text(corrupted, encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(final_path)


def test_manifest_generation_is_deterministic_for_same_inputs(tmp_path):
    source_root = tmp_path / "reports"
    _write_complete_evidence(source_root)
    first = begin_run(tmp_path / "first")
    second = begin_run(tmp_path / "second")

    first_manifest = build_manifest(
        first,
        stage_artifacts(first, source_root=source_root),
        required_types=ArtifactType,
    )
    second_manifest = build_manifest(
        second,
        stage_artifacts(second, source_root=source_root),
        required_types=ArtifactType,
    )
    first_artifacts = [item.model_dump() for item in first_manifest.artifacts]
    second_artifacts = [item.model_dump() for item in second_manifest.artifacts]

    assert first_artifacts == second_artifacts
    assert first_manifest.aggregate_byte_count == second_manifest.aggregate_byte_count
    assert first_manifest.aggregate_record_count == second_manifest.aggregate_record_count


def test_source_symlink_escape_is_rejected(tmp_path):
    source_root = tmp_path / "reports"
    outside = tmp_path / "outside"
    _write_trace_report(outside)

    try:
        (source_root).mkdir()
        (source_root / "observability").symlink_to(
            outside / "observability",
            target_is_directory=True,
        )
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    run = begin_run(tmp_path / "lifecycle")
    with pytest.raises(ValueError, match="escapes"):
        stage_artifacts(
            run,
            source_root=source_root,
            artifact_types=[ArtifactType.TRACES],
        )


def test_finalize_is_atomic_and_keeps_staging_when_publication_fails(tmp_path):
    source_root = tmp_path / "reports"
    lifecycle_root = tmp_path / "lifecycle"
    _write_trace_report(source_root)
    run = begin_run(lifecycle_root)
    artifacts = stage_artifacts(
        run,
        source_root=source_root,
        artifact_types=[ArtifactType.TRACES],
    )
    staged_artifact = run.staging_path / "artifacts" / "observability" / "traces.json"
    staged_artifact.unlink()

    with pytest.raises(FileNotFoundError):
        finalize_run(run, artifacts, required_types=[ArtifactType.TRACES])

    runs_root = lifecycle_root.resolve() / "runs"
    assert not (runs_root / run.run_id).exists()
    assert run.staging_path.exists()
    assert list(runs_root.glob(f".{run.run_id}.*.tmp")) == []


def test_concurrent_local_reservations_are_isolated(tmp_path):
    lifecycle_root = tmp_path / "lifecycle"

    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(lambda _: begin_run(lifecycle_root), range(8)))

    run_ids = [run.run_id for run in runs]
    assert len(run_ids) == len(set(run_ids))
    assert sorted(run_ids) == [f"local-{index:06d}" for index in range(1, 9)]
    assert all(run.staging_path.is_dir() for run in runs)


def test_legacy_mode_allows_empty_optional_evidence_bundle(tmp_path):
    final_path = create_bundle_from_existing_reports(
        source_root=tmp_path / "empty-reports",
        lifecycle_root=tmp_path / "lifecycle",
    )
    manifest = verify_bundle(final_path)

    assert manifest.artifacts == []
    assert manifest.aggregate_byte_count == 0
    assert manifest.aggregate_record_count == 0


def test_step1_to_step6_artifacts_are_ignored_by_lifecycle_bundling(tmp_path):
    source_root = tmp_path / "reports"
    _write_trace_report(source_root)
    ignored = [
        source_root / "api" / "phase1.json",
        source_root / "ui" / "phase3.html",
        source_root / "performance" / "phase6.json",
        source_root / "security" / "phase7.json",
    ]
    for path in ignored:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"ignored": true}', encoding="utf-8")

    run = begin_run(tmp_path / "lifecycle")
    artifacts = stage_artifacts(run, source_root=source_root)
    final_path = finalize_run(run, artifacts)
    manifest = verify_bundle(final_path)

    assert [item.artifact_type for item in manifest.artifacts] == [ArtifactType.TRACES]
    assert not any("phase" in item.relative_path for item in manifest.artifacts)


def test_retention_prunes_only_verified_bundles_after_confirmation(tmp_path):
    first = _create_trace_bundle(tmp_path, 1)
    second = _create_trace_bundle(tmp_path, 2)
    invalid = first.parent / "local-999999"
    invalid.mkdir()
    (invalid / "manifest.json").write_text("{}", encoding="utf-8")

    os.utime(first, (1, 1))
    os.utime(second, (2, 2))

    planned = plan_retention(tmp_path / "lifecycle", keep=1)
    dry_run = prune_runs(tmp_path / "lifecycle", keep=1, confirm=False)
    pruned = prune_runs(tmp_path / "lifecycle", keep=1, confirm=True)

    assert planned == [first]
    assert dry_run == [first]
    assert pruned == [first]
    assert not first.exists()
    assert second.exists()
    assert invalid.exists()


def test_retention_refuses_to_delete_paths_outside_runs_root(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(lifecycle, "plan_retention", lambda *args, **kwargs: [outside])

    with pytest.raises(ValueError, match="escapes"):
        prune_runs(tmp_path / "lifecycle", keep=1, confirm=True)

    assert outside.exists()
