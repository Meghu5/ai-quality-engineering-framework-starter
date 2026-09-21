from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from observability.context import (
    clear_context,
    generate_correlation_id,
    generate_trace_id,
    get_context,
    get_correlation_id,
    get_span_id,
    get_trace_id,
    set_correlation_id,
    use_context,
)


pytestmark = pytest.mark.observability


def test_context_starts_empty():
    assert get_context() is None


def test_generates_valid_trace_and_correlation_ids():
    trace_id = generate_trace_id()
    correlation_id = generate_correlation_id()

    assert len(trace_id) == 32
    assert int(trace_id, 16) > 0
    assert len(correlation_id) == 32


def test_explicit_correlation_id_is_propagated_and_reset():
    with use_context(correlation_id="booking-123"):
        assert get_correlation_id() == "booking-123"
        assert get_trace_id() is not None

    assert get_context() is None


def test_nested_context_preserves_parent_and_restores_it():
    with use_context(span_id="1" * 16) as parent:
        with use_context(span_id="2" * 16) as child:
            assert child.trace_id == parent.trace_id
            assert child.correlation_id == parent.correlation_id
            assert get_span_id() == "2" * 16
        assert get_span_id() == "1" * 16


def test_set_and_clear_context():
    set_correlation_id("external-correlation")
    assert get_correlation_id() == "external-correlation"
    clear_context()
    assert get_context() is None


def test_context_is_async_task_local():
    async def worker(correlation_id: str) -> tuple[str | None, str | None]:
        with use_context(correlation_id=correlation_id):
            await asyncio.sleep(0)
            return get_correlation_id(), get_trace_id()

    async def run_workers():
        return await asyncio.gather(worker("first"), worker("second"))

    with ThreadPoolExecutor(max_workers=1) as executor:
        first, second = executor.submit(asyncio.run, run_workers()).result()
    assert first[0] == "first"
    assert second[0] == "second"
    assert first[1] != second[1]
    assert get_context() is None
