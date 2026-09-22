from __future__ import annotations

import logging
from pathlib import Path
import shutil

import pytest

from observability.ci import (
    TestCorrelationState,
    begin_test_correlation,
    build_test_evidence,
    classify_test_exception,
    end_test_correlation,
    load_test_evidence,
    load_test_spans,
    worker_id_from_environment,
    write_ci_reports,
    write_test_evidence,
    write_test_spans,
    write_trace_report,
)


STATE_KEY = pytest.StashKey[TestCorrelationState]()
TOKEN_KEY = pytest.StashKey[object]()
EVIDENCE_KEY = pytest.StashKey[object]()
REPORT_ROOT_KEY = pytest.StashKey[Path]()
logger = logging.getLogger(__name__)


def pytest_configure(config: pytest.Config) -> None:
    root = Path("reports") / "ci"
    config.stash[REPORT_ROOT_KEY] = root
    try:
        if not hasattr(config, "workerinput"):
            workers = root / "workers"
            if workers.exists():
                shutil.rmtree(workers)
            for artifact in (
                root / "test-trace-correlation.json",
                root / "failure-summary.json",
            ):
                artifact.unlink(missing_ok=True)
    except OSError:
        logger.exception("Unable to prepare CI correlation evidence directory")


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_setup(item: pytest.Item):
    yield
    try:
        state, token = begin_test_correlation(item.nodeid)
        item.stash[STATE_KEY] = state
        item.stash[TOKEN_KEY] = token
    except Exception:
        logger.exception("Unable to establish pytest correlation context")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    state = item.stash.get(STATE_KEY, None)
    if state is None:
        return
    phase_outcome = report.outcome
    if hasattr(report, "wasxfail"):
        phase_outcome = "xfailed" if report.skipped else "xpassed"
    state.phase_outcomes[report.when] = phase_outcome
    state.duration_ms += report.duration * 1000
    if report.failed and state.test_failure_category is None:
        state.test_failure_category = classify_test_exception(
            call.excinfo.type if call.excinfo is not None else None
        )
    if report.when == "teardown":
        try:
            evidence = build_test_evidence(state)
            item.stash[EVIDENCE_KEY] = evidence
            root = item.config.stash[REPORT_ROOT_KEY]
            worker_id = worker_id_from_environment()
            write_test_evidence(evidence, root, worker_id)
            write_test_spans(state.spans, state.test_id, root, worker_id)
        except Exception:
            logger.exception("Unable to write pytest correlation evidence")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown(item: pytest.Item):
    yield
    state = item.stash.get(STATE_KEY, None)
    token = item.stash.get(TOKEN_KEY, None)
    if state is None or token is None:
        return
    try:
        end_test_correlation(token)
    except Exception:
        logger.exception("Unable to clear pytest correlation context")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if hasattr(session.config, "workerinput"):
        return
    try:
        root = session.config.stash[REPORT_ROOT_KEY]
        tests = load_test_evidence(root)
        write_ci_reports(tests)
        write_trace_report(load_test_spans(root))
    except Exception:
        logger.exception("Unable to finalize CI correlation evidence")
