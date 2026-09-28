from __future__ import annotations

from dataclasses import dataclass

from ai_quality.dataset import (
    RegisteredDatasetBundle,
    _canonicalize_registered_dataset,
)
from ai_quality.evaluator_registry import (
    RegisteredEvaluator,
    _canonicalize_registered_evaluator,
)
from ai_quality.models import AirlineAssistantResponse, GoldenCase, QualityReport
from ai_quality.prompt_registry import (
    RegisteredPrompt,
    _canonicalize_registered_prompt,
)
from ai_quality.provider_capabilities import (
    AIRLINE_RESPONSE_SCHEMA_ID,
    AIRLINE_RESPONSE_SCHEMA_VERSION,
    AIRLINE_RESPONSE_VALIDATOR_ID,
    AIRLINE_RESPONSE_VALIDATOR_VERSION,
    ProviderCapabilityRequirement,
)
from ai_quality.provider_config import RealLLMProviderSettings


@dataclass(frozen=True)
class SafeProviderExecutionSettings:
    retry_policy_id: str
    timeout_seconds: float | None
    max_attempts: int | None
    max_retry_delay_seconds: float | None
    structured_output_required: bool
    provider_required: bool


@dataclass(frozen=True)
class _FixedContractConstants:
    contract_version: str = "2.0"
    response_schema_id: str = AIRLINE_RESPONSE_SCHEMA_ID
    response_schema_version: str = AIRLINE_RESPONSE_SCHEMA_VERSION
    validator_id: str = AIRLINE_RESPONSE_VALIDATOR_ID
    validator_version: str = AIRLINE_RESPONSE_VALIDATOR_VERSION
    capability_contract_id: str = "provider-capability-contract"
    capability_contract_version: str = "1.0"


_CONTRACT_CONSTANTS = _FixedContractConstants()


class TrustedExecutionDescriptor:
    __slots__ = (
        "_prompt",
        "_dataset",
        "_evaluator",
        "_provider_settings",
        "_provider_id",
        "_model_id",
        "_capability_requirement",
        "_contract",
    )

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "trusted execution descriptors must be created from registered handles"
        )

    @classmethod
    def create(
        cls,
        *,
        registered_prompt: RegisteredPrompt,
        registered_dataset: RegisteredDatasetBundle,
        registered_evaluator: RegisteredEvaluator,
        provider_settings: RealLLMProviderSettings | None,
        provider_required: bool,
        provider_id: str,
        model_id: str,
    ) -> TrustedExecutionDescriptor:
        if not isinstance(registered_prompt, RegisteredPrompt) or not _is_issued(
            registered_prompt
        ):
            raise ValueError("trusted descriptor requires an enrolled prompt")
        if not isinstance(
            registered_dataset, RegisteredDatasetBundle
        ) or not _is_issued_bundle(registered_dataset):
            raise ValueError("trusted descriptor requires an enrolled dataset")
        if not isinstance(
            registered_evaluator, RegisteredEvaluator
        ) or not _is_issued_evaluator(registered_evaluator):
            raise ValueError("trusted descriptor requires an enrolled evaluator")
        if not isinstance(provider_required, bool):
            raise TypeError("provider_required must be a boolean")
        safe_settings = _safe_provider_settings(
            provider_settings,
            provider_required=provider_required,
        )
        if not isinstance(provider_id, str) or not provider_id:
            raise ValueError("provider_id must be a safe non-empty identifier")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a safe non-empty identifier")
        descriptor = object.__new__(cls)
        object.__setattr__(descriptor, "_prompt", registered_prompt)
        object.__setattr__(descriptor, "_dataset", registered_dataset)
        object.__setattr__(descriptor, "_evaluator", registered_evaluator)
        object.__setattr__(descriptor, "_provider_settings", safe_settings)
        object.__setattr__(descriptor, "_provider_id", provider_id)
        object.__setattr__(descriptor, "_model_id", model_id)
        object.__setattr__(
            descriptor,
            "_capability_requirement",
            ProviderCapabilityRequirement(),
        )
        object.__setattr__(descriptor, "_contract", _CONTRACT_CONSTANTS)
        return descriptor

    @property
    def prompt(self) -> RegisteredPrompt:
        return self._prompt

    @property
    def dataset(self) -> RegisteredDatasetBundle:
        return self._dataset

    @property
    def evaluator(self) -> RegisteredEvaluator:
        return self._evaluator

    @property
    def provider_settings(self) -> SafeProviderExecutionSettings:
        return self._provider_settings

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capability_requirement(self) -> ProviderCapabilityRequirement:
        return self._capability_requirement

    @property
    def contract(self) -> _FixedContractConstants:
        return self._contract

    def materialize_cases(self) -> list[GoldenCase]:
        return self._dataset.materialize_cases()

    def evaluate(
        self,
        cases: list[GoldenCase],
        responses: list[AirlineAssistantResponse],
    ) -> QualityReport:
        return self._evaluator.evaluate(cases, responses)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("trusted execution descriptors are immutable")

    def __repr__(self) -> str:
        return (
            "TrustedExecutionDescriptor("
            f"prompt={self.prompt.prompt.name!r}/{self.prompt.prompt.version!r}, "
            f"dataset={self.dataset.dataset_id!r}/{self.dataset.dataset_version!r}, "
            f"evaluator={self.evaluator.evaluator_id!r}/"
            f"{self.evaluator.evaluator_version!r}, "
            f"provider_id={self.provider_id!r}, model_id={self.model_id!r})"
        )


def _safe_provider_settings(
    settings: RealLLMProviderSettings | None,
    *,
    provider_required: bool,
) -> SafeProviderExecutionSettings:
    if settings is None:
        return SafeProviderExecutionSettings(
            retry_policy_id="provider-managed",
            timeout_seconds=None,
            max_attempts=None,
            max_retry_delay_seconds=None,
            structured_output_required=True,
            provider_required=provider_required,
        )
    if not isinstance(settings, RealLLMProviderSettings):
        raise TypeError("provider settings must be RealLLMProviderSettings")
    settings.validate()
    if settings.required != provider_required:
        raise ValueError("provider settings conflict with effective execution policy")
    return SafeProviderExecutionSettings(
        retry_policy_id="bounded-http-retry",
        timeout_seconds=float(settings.timeout_seconds),
        max_attempts=settings.max_attempts,
        max_retry_delay_seconds=float(settings.max_retry_delay_seconds),
        structured_output_required=settings.require_structured_output,
        provider_required=provider_required,
    )


def _bind_descriptor_creation(
    descriptor_type: type[TrustedExecutionDescriptor],
):
    prompt_canonicalizer = _canonicalize_registered_prompt
    dataset_canonicalizer = _canonicalize_registered_dataset
    evaluator_canonicalizer = _canonicalize_registered_evaluator
    settings_projector = _safe_provider_settings
    contract = _CONTRACT_CONSTANTS

    def create(
        cls,
        *,
        registered_prompt: RegisteredPrompt,
        registered_dataset: RegisteredDatasetBundle,
        registered_evaluator: RegisteredEvaluator,
        provider_settings: RealLLMProviderSettings | None,
        provider_required: bool,
        provider_id: str,
        model_id: str,
    ) -> TrustedExecutionDescriptor:
        canonical_prompt = prompt_canonicalizer(registered_prompt)
        canonical_dataset = dataset_canonicalizer(registered_dataset)
        canonical_evaluator = evaluator_canonicalizer(registered_evaluator)
        if not isinstance(provider_required, bool):
            raise TypeError("provider_required must be a boolean")
        safe_settings = settings_projector(
            provider_settings,
            provider_required=provider_required,
        )
        if not isinstance(provider_id, str) or not provider_id:
            raise ValueError("provider_id must be a safe non-empty identifier")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a safe non-empty identifier")
        descriptor = object.__new__(descriptor_type)
        object.__setattr__(descriptor, "_prompt", canonical_prompt)
        object.__setattr__(descriptor, "_dataset", canonical_dataset)
        object.__setattr__(descriptor, "_evaluator", canonical_evaluator)
        object.__setattr__(descriptor, "_provider_settings", safe_settings)
        object.__setattr__(descriptor, "_provider_id", provider_id)
        object.__setattr__(descriptor, "_model_id", model_id)
        object.__setattr__(
            descriptor,
            "_capability_requirement",
            ProviderCapabilityRequirement(),
        )
        object.__setattr__(descriptor, "_contract", contract)
        return descriptor

    return classmethod(create)


TrustedExecutionDescriptor.create = _bind_descriptor_creation(
    TrustedExecutionDescriptor
)
del _bind_descriptor_creation
