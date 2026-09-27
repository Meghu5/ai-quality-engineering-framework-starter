from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict, Field

from ai_quality.models import GoldenCase


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "ai" / "golden_cases.json"
DATASET_MANIFEST_PATH = DATASET_PATH.with_name("golden_cases_manifest.json")
UNREGISTERED_DATASET_ID = "unregistered"
UNREGISTERED_DATASET_VERSION = "0"
REGISTERED_DATASET_ID = "airline-quality-golden-cases"
REGISTERED_DATASET_VERSION = "1.0"
_REGISTERED_DATASET_INTEGRITY = (
    "88d258e382112f5d024e21712bd3ece805925383797fc19183fa3282507691d8"
)
_REGISTERED_CASE_IDS = (
    "ai-001-flight-search", "ai-002-flight-status", "ai-003-booking",
    "ai-004-cancellation", "ai-005-baggage", "ai-006-check-in",
    "ai-007-seat-selection", "ai-008-fare", "ai-009-refund",
    "ai-010-general-help", "ai-011-unsupported", "ai-012-ambiguous",
    "ai-013-missing-origin", "ai-014-invalid-airport", "ai-015-invalid-date",
    "ai-016-multiple-entities", "ai-017-multi-turn",
    "ai-018-safety-sensitive", "ai-019-prompt-injection", "ai-020-pii",
    "ai-021-out-of-domain", "ai-022-hallucination-trap",
    "ai-023-unsupported-policy", "ai-024-tool-action",
    "ai-025-structured-output",
)


@dataclass(frozen=True)
class _RegisteredDatasetDefinition:
    case_revisions: tuple[tuple[str, str], ...]
    integrity_digest: str


class GoldenCaseRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    case_id: str
    revision: str = Field(pattern=r"(?:v)?[0-9]+(?:\.[0-9]+){0,3}")


class GoldenDatasetIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dataset_id: str = Field(pattern=r"[a-z][a-z0-9-]{0,63}")
    dataset_version: str = Field(pattern=r"(?:v)?[0-9]+(?:\.[0-9]+){0,3}")
    cases: tuple[GoldenCaseRevision, ...]


class RegisteredDatasetBundle:
    __slots__ = (
        "_identity",
        "_canonical_source",
        "_capability",
        "__weakref__",
    )

    def __new__(cls, *args, **kwargs):
        raise TypeError("registered datasets must be resolved by DatasetRegistry")

    @property
    def dataset_id(self) -> str:
        return self._identity.dataset_id

    @property
    def dataset_version(self) -> str:
        return self._identity.dataset_version

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(item.case_id for item in self._identity.cases)

    @property
    def case_revisions(self) -> tuple[str, ...]:
        return tuple(item.revision for item in self._identity.cases)

    def materialize_cases(self) -> list[GoldenCase]:
        if not _is_issued_bundle(self):
            raise ValueError("registered dataset capability is invalid")
        raw_cases = json.loads(self._canonical_source)
        return [GoldenCase.model_validate(case) for case in raw_cases]

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("registered datasets are immutable")

    def __repr__(self) -> str:
        return (
            "RegisteredDatasetBundle("
            f"dataset_id={self.dataset_id!r}, "
            f"dataset_version={self.dataset_version!r}, "
            f"case_count={len(self.case_ids)})"
        )


_ISSUED_DATASETS: WeakKeyDictionary[RegisteredDatasetBundle, object] = (
    WeakKeyDictionary()
)
_REGISTERED_DATASETS = MappingProxyType(
    {
        (REGISTERED_DATASET_ID, REGISTERED_DATASET_VERSION): (
            _RegisteredDatasetDefinition(
                case_revisions=tuple((case_id, "1") for case_id in _REGISTERED_CASE_IDS),
                integrity_digest=_REGISTERED_DATASET_INTEGRITY,
            )
        )
    }
)


class DatasetRegistry:
    def resolve(self, name: str, version: str) -> RegisteredDatasetBundle:
        registration = _REGISTERED_DATASETS.get((name, version))
        if registration is None:
            raise KeyError("dataset registration not found")
        manifest = load_golden_dataset_identity()
        if manifest.dataset_id != name or manifest.dataset_version != version:
            raise ValueError("registered dataset manifest identity is inconsistent")
        manifest_revisions = tuple(
            (item.case_id, item.revision) for item in manifest.cases
        )
        if manifest_revisions != registration.case_revisions:
            raise ValueError("registered dataset revision declaration is inconsistent")
        raw_cases = _load_raw_cases(DATASET_PATH)
        if tuple(case["id"] for case in raw_cases) != tuple(
            item.case_id for item in manifest.cases
        ):
            raise ValueError("registered dataset case order is inconsistent")
        actual_integrity = _dataset_integrity_digest(manifest, raw_cases)
        if actual_integrity != registration.integrity_digest:
            raise ValueError("registered dataset integrity validation failed")
        canonical_source = _canonical_json(raw_cases)
        return _issue_registered_dataset(manifest, canonical_source)


def load_golden_dataset_identity(
    path: Path | None = None,
) -> GoldenDatasetIdentity:
    with (path or DATASET_MANIFEST_PATH).open(encoding="utf-8") as file:
        raw = json.load(file)
    if set(raw) != {"dataset_id", "dataset_version", "cases"}:
        raise ValueError("golden dataset manifest contains unsupported fields")
    entries = raw["cases"]
    if not isinstance(entries, list):
        raise ValueError("golden dataset manifest cases must be a list")
    revisions: list[GoldenCaseRevision] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"case_id", "revision"}:
            raise ValueError("golden dataset manifest case is invalid")
        case_id = entry["case_id"]
        revision = entry["revision"]
        if not isinstance(case_id, str) or not isinstance(revision, str):
            raise ValueError("golden dataset manifest values must be strings")
        revisions.append(GoldenCaseRevision(case_id=case_id, revision=revision))
    return GoldenDatasetIdentity(
        dataset_id=raw["dataset_id"],
        dataset_version=raw["dataset_version"],
        cases=tuple(revisions),
    )


def resolve_golden_dataset_identity(cases: list[GoldenCase]) -> GoldenDatasetIdentity:
    return GoldenDatasetIdentity(
        dataset_id=UNREGISTERED_DATASET_ID,
        dataset_version=UNREGISTERED_DATASET_VERSION,
        cases=tuple(
            GoldenCaseRevision(case_id=case.id, revision="0") for case in cases
        ),
    )


def load_golden_cases(path: Path = DATASET_PATH) -> list[GoldenCase]:
    raw_cases = _load_raw_cases(path)
    return [GoldenCase.model_validate(case) for case in raw_cases]


def dataset_for_execution(
    value: list[GoldenCase] | RegisteredDatasetBundle,
) -> tuple[list[GoldenCase], GoldenDatasetIdentity]:
    if isinstance(value, RegisteredDatasetBundle):
        if not _is_issued_bundle(value):
            raise ValueError("registered dataset capability is invalid")
        return value.materialize_cases(), value._identity
    if isinstance(value, list) and all(isinstance(case, GoldenCase) for case in value):
        return value, resolve_golden_dataset_identity(value)
    raise TypeError("cases must be a list of GoldenCase or RegisteredDatasetBundle")


def _issue_registered_dataset(
    identity: GoldenDatasetIdentity,
    canonical_source: str,
) -> RegisteredDatasetBundle:
    bundle = object.__new__(RegisteredDatasetBundle)
    object.__setattr__(bundle, "_identity", identity)
    object.__setattr__(bundle, "_canonical_source", canonical_source)
    capability = object()
    object.__setattr__(bundle, "_capability", capability)
    _ISSUED_DATASETS[bundle] = capability
    return bundle


def _is_issued_bundle(bundle: RegisteredDatasetBundle) -> bool:
    return _ISSUED_DATASETS.get(bundle) is bundle._capability


def _load_raw_cases(path: Path) -> list[dict[str, object]]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list) or any(
        not isinstance(case, dict) or not isinstance(case.get("id"), str)
        for case in raw_cases
    ):
        raise ValueError("golden dataset source is invalid")
    return raw_cases


def _dataset_integrity_digest(
    manifest: GoldenDatasetIdentity,
    raw_cases: list[dict[str, object]],
) -> str:
    projection = {
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "case_revisions": [case.model_dump(mode="json") for case in manifest.cases],
        "case_definitions": raw_cases,
    }
    return hashlib.sha256(_canonical_json(projection).encode("ascii")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
