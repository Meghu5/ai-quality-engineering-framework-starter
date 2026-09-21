from __future__ import annotations

import pytest

from observability.config import ObservabilitySettings
from observability.context import clear_context
from observability.exporters import InMemoryExporter
from observability.tracing import TracingFacade


@pytest.fixture(autouse=True)
def isolated_trace_context():
    clear_context()
    yield
    clear_context()


@pytest.fixture
def memory_exporter():
    return InMemoryExporter()


@pytest.fixture
def tracing(memory_exporter):
    settings = ObservabilitySettings(enabled=True, exporter="memory")
    return TracingFacade(settings=settings, exporter=memory_exporter)
