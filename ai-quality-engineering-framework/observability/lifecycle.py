from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ai_eval.models import Phase10Report
from observability.analysis import OperationalAnalysisReport
from observability.ci import CiCorrelationReport
from observability.models import TraceEnvelope
from observability.readiness import ObservabilityReadinessReport


logger = logging.getLogger(__name__)
MANIFEST_SCHEMA_VERSION = "1.0"
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset({MANIFEST_SCHEMA_VERSION})
PRODUCER_VERSION = "phase11-step7"
DEFAULT_LIFECYCLE_ROOT = Path("reports") / "observability" / "lifecycle"
DEFAULT_SOURCE_ROOT = Path("reports")
MAX_ARTIFACT_BYTES = 25 * 1024 * 1024
MAX_BUNDLE_BYTES = 100 * 1024 * 1024
_RUN_ID = re.compile(r"(?:ci-[0-9a-f]{24}|explicit-[0-9a-f]{24}|local-[0-9]{6,})")
_CI_NUMBER = re.compile(r"[0-9]{1,32}")
_CI_JOB = re.compile(r"[A-Za-z0-9_.-]{1,100}")


class ArtifactType(str, Enum):
    TRACES = "traces"
    CI_CORRELATION = "ci_correlation"
    FAILURE_SUMMARY = "failure_summary"
    PHASE10_REPORT = "phase10_report"
    OPERATIONAL_ANALYSIS = "operational_analysis"
    READINESS = "readiness"


class RunIdSource(str, Enum):
    CI = "ci"
    EXPLICIT = "explicit"
    LOCAL = "local"


class ManifestArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: ArtifactType
    relative_path: str
    schema_version: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=0)
    record_count: int = Field(ge=0)
    required: bool
    status: Literal["valid"] = "valid"


class EvidenceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(min_length=1, strict=True)
    producer: str = "ai-quality-engineering-framework"
    producer_version: str = PRODUCER_VERSION
    run_id: str
    run_id_source: RunIdSource
    bundle_status: Literal["complete"] = "complete"
    artifacts: list[ManifestArtifact]
    aggregate_byte_count: int = Field(ge=0)
    aggregate_record_count: int = Field(ge=0)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
            raise ValueError("unsupported manifest schema version")
        return value


class LifecycleRun(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_id: str
    run_id_source: RunIdSource
    lifecycle_root: Path
    staging_path: Path


class ArtifactSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    artifact_type: ArtifactType
    relative_path: Path
    schema_version: str
    model_type: type[BaseModel] | None = None
    collection: bool = False


ARTIFACT_SPECS: dict[ArtifactType, ArtifactSpec] = {
    ArtifactType.TRACES: ArtifactSpec(
        artifact_type=ArtifactType.TRACES,
        relative_path=Path("observability") / "traces.json",
        schema_version="1.0",
        model_type=TraceEnvelope,
        collection=True,
    ),
    ArtifactType.CI_CORRELATION: ArtifactSpec(
        artifact_type=ArtifactType.CI_CORRELATION,
        relative_path=Path("ci") / "test-trace-correlation.json",
        schema_version="1.0",
        model_type=CiCorrelationReport,
    ),
    ArtifactType.FAILURE_SUMMARY: ArtifactSpec(
        artifact_type=ArtifactType.FAILURE_SUMMARY,
        relative_path=Path("ci") / "failure-summary.json",
        schema_version="1.0",
    ),
    ArtifactType.PHASE10_REPORT: ArtifactSpec(
        artifact_type=ArtifactType.PHASE10_REPORT,
        relative_path=Path("ai_eval") / "phase10_report.json",
        schema_version="1.0",
        model_type=Phase10Report,
    ),
    ArtifactType.OPERATIONAL_ANALYSIS: ArtifactSpec(
        artifact_type=ArtifactType.OPERATIONAL_ANALYSIS,
        relative_path=Path("observability") / "operational-analysis.json",
        schema_version="1.0",
        model_type=OperationalAnalysisReport,
    ),
    ArtifactType.READINESS: ArtifactSpec(
        artifact_type=ArtifactType.READINESS,
        relative_path=Path("observability") / "readiness.json",
        schema_version="1.0",
        model_type=ObservabilityReadinessReport,
    ),
}

SUPPORTED_SCHEMA_VERSIONS: dict[ArtifactType, frozenset[str]] = {
    artifact_type: frozenset({spec.schema_version})
    for artifact_type, spec in ARTIFACT_SPECS.items()
}


def begin_run(
    lifecycle_root: Path = DEFAULT_LIFECYCLE_ROOT,
    *,
    environment: dict[str, str] | None = None,
) -> LifecycleRun:
    env = environment if environment is not None else dict(os.environ)
    root = _resolved_root(lifecycle_root)
    staging_root = root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)

    explicit = env.get("AI_OBSERVABILITY_RUN_ID", "").strip()
    if explicit:
        run_id = _hashed_run_id("explicit", explicit)
        return _reserve_run(run_id, RunIdSource.EXPLICIT, root, staging_root)

    github_run = env.get("GITHUB_RUN_ID", "").strip()
    if github_run:
        attempt = env.get("GITHUB_RUN_ATTEMPT", "1").strip()
        job = env.get("GITHUB_JOB", "observability").strip()
        if not _CI_NUMBER.fullmatch(github_run) or not _CI_NUMBER.fullmatch(attempt):
            raise ValueError("GitHub run identity must contain decimal digits only")
        if not _CI_JOB.fullmatch(job):
            raise ValueError("GitHub job identity contains unsupported characters")
        run_id = _hashed_run_id("ci", f"{github_run}:{attempt}:{job}")
        return _reserve_run(run_id, RunIdSource.CI, root, staging_root)

    index = 1
    while True:
        run_id = f"local-{index:06d}"
        try:
            return _reserve_run(run_id, RunIdSource.LOCAL, root, staging_root)
        except FileExistsError:
            index += 1


def stage_artifacts(
    run: LifecycleRun,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    artifact_types: Iterable[ArtifactType] | None = None,
) -> list[ManifestArtifact]:
    _validate_run(run)
    selected = sorted(
        artifact_types or ARTIFACT_SPECS,
        key=lambda item: item.value,
    )
    staged: list[ManifestArtifact] = []
    aggregate = 0
    for artifact_type in selected:
        spec = ARTIFACT_SPECS[artifact_type]
        source = _safe_join(source_root, spec.relative_path)
        if not source.is_file():
            continue
        source = source.resolve()
        byte_count = source.stat().st_size
        if byte_count > MAX_ARTIFACT_BYTES:
            raise ValueError(f"{artifact_type.value} exceeds the artifact size limit")
        aggregate += byte_count
        if aggregate > MAX_BUNDLE_BYTES:
            raise ValueError("evidence bundle exceeds the aggregate size limit")
        record_count = validate_artifact(source, artifact_type, spec.schema_version)
        destination = _safe_join(run.staging_path / "artifacts", spec.relative_path)
        _copy_file_atomic(source, destination, MAX_ARTIFACT_BYTES)
        staged.append(
            ManifestArtifact(
                artifact_type=artifact_type,
                relative_path=spec.relative_path.as_posix(),
                schema_version=spec.schema_version,
                sha256=hash_file(destination),
                byte_count=byte_count,
                record_count=record_count,
                required=False,
            )
        )
    return staged


def build_manifest(
    run: LifecycleRun,
    artifacts: list[ManifestArtifact],
    *,
    required_types: Iterable[ArtifactType] = (),
) -> EvidenceManifest:
    required = set(required_types)
    present = {item.artifact_type for item in artifacts}
    missing = sorted(required - present, key=lambda item: item.value)
    if missing:
        raise ValueError(
            "required lifecycle artifacts are missing: "
            + ", ".join(item.value for item in missing)
        )
    ordered = [
        item.model_copy(update={"required": item.artifact_type in required})
        for item in sorted(artifacts, key=lambda item: item.relative_path)
    ]
    return EvidenceManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=run.run_id,
        run_id_source=run.run_id_source,
        artifacts=ordered,
        aggregate_byte_count=sum(item.byte_count for item in ordered),
        aggregate_record_count=sum(item.record_count for item in ordered),
    )


def finalize_run(
    run: LifecycleRun,
    artifacts: list[ManifestArtifact],
    *,
    required_types: Iterable[ArtifactType] = (),
) -> Path:
    _validate_run(run)
    manifest = build_manifest(run, artifacts, required_types=required_types)
    runs_root = run.lifecycle_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    final_path = runs_root / run.run_id
    if final_path.exists():
        raise FileExistsError(f"finalized lifecycle run already exists: {run.run_id}")
    temporary_path = runs_root / f".{run.run_id}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    temporary_path.mkdir(exist_ok=False)
    try:
        for item in manifest.artifacts:
            source = _safe_join(
                run.staging_path / "artifacts", Path(item.relative_path)
            )
            destination = _safe_join(
                temporary_path / "artifacts", Path(item.relative_path)
            )
            _copy_file_atomic(source, destination, MAX_ARTIFACT_BYTES)
        _write_json_atomic(
            temporary_path / "manifest.json", manifest.model_dump(mode="json")
        )
        _verify_bundle(temporary_path, require_matching_directory=False)
        temporary_path.rename(final_path)
        verify_bundle(final_path)
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise
    shutil.rmtree(run.staging_path)
    return final_path


def verify_bundle(bundle_path: Path) -> EvidenceManifest:
    return _verify_bundle(bundle_path, require_matching_directory=True)


def _verify_bundle(
    bundle_path: Path, *, require_matching_directory: bool
) -> EvidenceManifest:
    manifest_path = _safe_join(bundle_path, Path("manifest.json"))
    if not manifest_path.is_file():
        raise ValueError("evidence bundle is incomplete: manifest is absent")
    if manifest_path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("manifest exceeds the artifact size limit")
    try:
        payload = _read_bounded_json(manifest_path)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("manifest is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("manifest must contain a JSON object")
    if "schema_version" not in payload:
        raise ValueError("manifest schema version is missing")
    version = payload["schema_version"]
    if not isinstance(version, str) or not version:
        raise ValueError("manifest schema version is malformed")
    if version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError("unsupported manifest schema version")
    try:
        manifest = EvidenceManifest.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("manifest schema validation failed") from exc
    if not _RUN_ID.fullmatch(manifest.run_id):
        raise ValueError("manifest run identity is invalid")
    if require_matching_directory and bundle_path.name != manifest.run_id:
        raise ValueError("manifest run identity does not match the bundle directory")
    aggregate_bytes = 0
    aggregate_records = 0
    seen: set[ArtifactType] = set()
    for item in manifest.artifacts:
        if item.artifact_type in seen:
            raise ValueError("manifest contains a duplicate artifact type")
        seen.add(item.artifact_type)
        spec = ARTIFACT_SPECS[item.artifact_type]
        if item.relative_path != spec.relative_path.as_posix():
            raise ValueError("manifest artifact path is not allowlisted")
        path = _safe_join(bundle_path / "artifacts", spec.relative_path)
        if not path.is_file() or path.stat().st_size != item.byte_count:
            raise ValueError("artifact byte count does not match the manifest")
        if hash_file(path) != item.sha256:
            raise ValueError("artifact checksum does not match the manifest")
        records = validate_artifact(path, item.artifact_type, item.schema_version)
        if records != item.record_count:
            raise ValueError("artifact record count does not match the manifest")
        aggregate_bytes += item.byte_count
        aggregate_records += item.record_count
    if aggregate_bytes != manifest.aggregate_byte_count:
        raise ValueError("bundle byte count does not match the manifest")
    if aggregate_records != manifest.aggregate_record_count:
        raise ValueError("bundle record count does not match the manifest")
    return manifest


def validate_artifact(path: Path, artifact_type: ArtifactType, schema_version: str) -> int:
    supported = SUPPORTED_SCHEMA_VERSIONS.get(artifact_type, frozenset())
    if schema_version not in supported:
        raise ValueError(f"unsupported schema version for {artifact_type.value}")
    payload = _read_bounded_json(path)
    spec = ARTIFACT_SPECS[artifact_type]
    if spec.collection:
        if not isinstance(payload, list):
            raise ValueError(f"{artifact_type.value} must contain a JSON array")
        for item in payload:
            spec.model_type.model_validate(item)
        return len(payload)
    if spec.model_type is not None:
        spec.model_type.model_validate(payload)
        return _model_record_count(payload, artifact_type)
    if artifact_type == ArtifactType.FAILURE_SUMMARY:
        if not isinstance(payload, dict):
            raise ValueError("failure summary must contain a JSON object")
        return int(payload.get("total_tests", 0))
    raise ValueError("artifact type has no validator")


def hash_file(path: Path, *, maximum_bytes: int = MAX_ARTIFACT_BYTES) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError("artifact exceeds the hashing size limit")
            digest.update(chunk)
    return digest.hexdigest()


def plan_retention(
    lifecycle_root: Path = DEFAULT_LIFECYCLE_ROOT, *, keep: int
) -> list[Path]:
    if keep < 0:
        raise ValueError("retention count must not be negative")
    root = _resolved_root(lifecycle_root)
    runs_root = root / "runs"
    if not runs_root.exists():
        return []
    valid: list[Path] = []
    for path in runs_root.iterdir():
        if not path.is_dir() or not _RUN_ID.fullmatch(path.name):
            continue
        try:
            verify_bundle(path)
        except (OSError, ValueError):
            continue
        valid.append(path)
    ordered = sorted(valid, key=lambda path: (path.stat().st_mtime_ns, path.name))
    return ordered[: max(0, len(ordered) - keep)]


def prune_runs(
    lifecycle_root: Path = DEFAULT_LIFECYCLE_ROOT,
    *,
    keep: int,
    confirm: bool = False,
) -> list[Path]:
    planned = plan_retention(lifecycle_root, keep=keep)
    if not confirm:
        return planned
    root = _resolved_root(lifecycle_root)
    runs_root = (root / "runs").resolve()
    for path in planned:
        resolved = path.resolve()
        if resolved.parent != runs_root:
            raise ValueError("retention target escapes the lifecycle runs directory")
        shutil.rmtree(resolved)
    return planned


def create_bundle_from_existing_reports(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    lifecycle_root: Path = DEFAULT_LIFECYCLE_ROOT,
    environment: dict[str, str] | None = None,
    required_types: Iterable[ArtifactType] = (),
) -> Path:
    run = begin_run(lifecycle_root, environment=environment)
    artifacts = stage_artifacts(run, source_root=source_root)
    return finalize_run(run, artifacts, required_types=required_types)


def _reserve_run(
    run_id: str,
    source: RunIdSource,
    root: Path,
    staging_root: Path,
) -> LifecycleRun:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("generated run identity is invalid")
    if (root / "runs" / run_id).exists():
        raise FileExistsError(f"finalized lifecycle run already exists: {run_id}")
    staging = staging_root / run_id
    staging.mkdir(exist_ok=False)
    return LifecycleRun(
        run_id=run_id,
        run_id_source=source,
        lifecycle_root=root,
        staging_path=staging,
    )


def _hashed_run_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _validate_run(run: LifecycleRun) -> None:
    root = _resolved_root(run.lifecycle_root)
    staging = run.staging_path.resolve()
    if not _RUN_ID.fullmatch(run.run_id):
        raise ValueError("run identity is invalid")
    if staging.parent != (root / ".staging").resolve() or staging.name != run.run_id:
        raise ValueError("staging path escapes the lifecycle root")
    if not staging.is_dir():
        raise ValueError("staging directory does not exist")


def _resolved_root(path: Path) -> Path:
    return path.resolve()


def _safe_join(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path must be a safe relative path")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError("artifact path escapes its configured root")
    return candidate


def _read_bounded_json(path: Path) -> Any:
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("artifact exceeds the input size limit")
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_file_atomic(source: Path, destination: Path, maximum_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(
        destination.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    total = 0
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while chunk := reader.read(64 * 1024):
                total += len(chunk)
                if total > maximum_bytes:
                    raise ValueError("artifact exceeds the copy size limit")
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _model_record_count(payload: Any, artifact_type: ArtifactType) -> int:
    if artifact_type == ArtifactType.CI_CORRELATION:
        return len(payload.get("tests", []))
    if artifact_type == ArtifactType.PHASE10_REPORT:
        return sum(len(item.get("results", [])) for item in payload.get("frameworks", []))
    return 1


def main() -> int:
    try:
        path = create_bundle_from_existing_reports()
        logger.info("Finalized observability evidence bundle %s", path.name)
        return 0
    except Exception as exc:
        logger.error("Unable to finalize observability evidence bundle (%s)", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
