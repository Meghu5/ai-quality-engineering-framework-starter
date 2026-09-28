from __future__ import annotations

import json

import pytest

import ai_quality.dataset as dataset_module
from ai_quality.dataset import (
    REGISTERED_DATASET_ID,
    REGISTERED_DATASET_VERSION,
    DatasetRegistry,
    GoldenDatasetIdentity,
    RegisteredDatasetBundle,
    dataset_for_execution,
    load_golden_dataset_identity,
)


def _resolve() -> RegisteredDatasetBundle:
    return DatasetRegistry().resolve(
        REGISTERED_DATASET_ID,
        REGISTERED_DATASET_VERSION,
    )


def test_resolve_returns_immutable_enrolled_bundle():
    bundle = _resolve()

    assert isinstance(bundle, RegisteredDatasetBundle)
    assert bundle.dataset_id == REGISTERED_DATASET_ID
    assert bundle.dataset_version == REGISTERED_DATASET_VERSION
    assert bundle.case_ids[0] == "ai-001-flight-search"
    assert bundle.case_revisions == ("1",) * 25
    with pytest.raises(AttributeError, match="immutable"):
        bundle.dataset_version = "2.0"
    with pytest.raises(TypeError, match="resolved by DatasetRegistry"):
        RegisteredDatasetBundle()


@pytest.mark.parametrize(
    ("name", "version"),
    [
        (REGISTERED_DATASET_ID, "2.0"),
        ("other-dataset", REGISTERED_DATASET_VERSION),
        (REGISTERED_DATASET_ID.upper(), REGISTERED_DATASET_VERSION),
    ],
)
def test_invalid_resolution_fails_deterministically(name, version):
    with pytest.raises(KeyError, match="registration not found"):
        DatasetRegistry().resolve(name, version)


def test_materialization_returns_fresh_copies_and_preserves_canonical_source():
    bundle = _resolve()
    first = bundle.materialize_cases()
    second = bundle.materialize_cases()

    assert first == second
    assert first is not second
    assert all(left is not right for left, right in zip(first, second))
    original = second[0].user_input
    first[0].user_input = "caller tampering"

    third = bundle.materialize_cases()
    assert third[0].user_input == original
    assert third[0].user_input != first[0].user_input


def test_matching_caller_cases_remain_unregistered():
    bundle = _resolve()
    caller_cases = [case.model_copy(deep=True) for case in bundle.materialize_cases()]

    execution_cases, identity = dataset_for_execution(caller_cases)

    assert execution_cases is caller_cases
    assert identity.dataset_id == "unregistered"
    assert identity.dataset_version == "0"
    assert tuple(case.id for case in caller_cases) == bundle.case_ids


def test_forged_bundle_capability_is_rejected():
    legitimate = _resolve()
    forged = object.__new__(RegisteredDatasetBundle)
    object.__setattr__(forged, "_identity", legitimate._identity)
    object.__setattr__(forged, "_canonical_source", legitimate._canonical_source)
    object.__setattr__(forged, "_enrollment_proof", object())

    with pytest.raises(ValueError, match="capability is invalid"):
        dataset_for_execution(forged)


def test_post_binding_dataset_path_replacement_cannot_redirect_enrollment(
    monkeypatch, tmp_path
):
    raw_cases = json.loads(dataset_module.DATASET_PATH.read_text(encoding="utf-8"))
    raw_cases[0]["user_input"] = "unauthorized changed prompt"
    changed = tmp_path / "changed-cases.json"
    changed.write_text(json.dumps(raw_cases), encoding="utf-8")
    monkeypatch.setattr(dataset_module, "DATASET_PATH", changed)

    assert _resolve().materialize_cases()[0].user_input != "unauthorized changed prompt"


def test_post_binding_reordered_dataset_path_is_ignored(monkeypatch, tmp_path):
    raw_cases = json.loads(dataset_module.DATASET_PATH.read_text(encoding="utf-8"))
    raw_cases[0], raw_cases[1] = raw_cases[1], raw_cases[0]
    changed = tmp_path / "reordered-cases.json"
    changed.write_text(json.dumps(raw_cases), encoding="utf-8")
    monkeypatch.setattr(dataset_module, "DATASET_PATH", changed)

    assert _resolve().case_ids[:2] == (
        "ai-001-flight-search",
        "ai-002-flight-status",
    )


def test_post_binding_manifest_path_replacement_cannot_redirect_revisions(
    monkeypatch, tmp_path
):
    manifest = load_golden_dataset_identity()
    changed_cases = list(manifest.cases)
    changed_cases[0] = changed_cases[0].model_copy(update={"revision": "2"})
    changed = manifest.model_copy(update={"cases": tuple(changed_cases)})
    changed_path = tmp_path / "changed-manifest.json"
    changed_path.write_text(changed.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(dataset_module, "DATASET_MANIFEST_PATH", changed_path)

    assert _resolve().case_revisions[0] == "1"


def test_post_binding_manifest_path_replacement_cannot_redirect_identity(
    monkeypatch, tmp_path
):
    manifest = load_golden_dataset_identity().model_copy(
        update={"dataset_version": "2.0"}
    )
    changed_path = tmp_path / "changed-manifest.json"
    changed_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(dataset_module, "DATASET_MANIFEST_PATH", changed_path)

    assert _resolve().dataset_version == REGISTERED_DATASET_VERSION


def test_repr_and_public_projection_exclude_source_digest_and_capability():
    bundle = _resolve()
    public = json.dumps(
        {
            "dataset_id": bundle.dataset_id,
            "dataset_version": bundle.dataset_version,
            "case_ids": bundle.case_ids,
            "case_revisions": bundle.case_revisions,
        },
        sort_keys=True,
    )

    for forbidden in (
        dataset_module._REGISTERED_DATASET_INTEGRITY,
        str(dataset_module.DATASET_PATH),
        "_canonical_source",
        "_capability",
    ):
        assert forbidden not in public
        assert forbidden not in repr(bundle)
