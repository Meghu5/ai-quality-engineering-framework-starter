from __future__ import annotations

from playwright.sync_api import Page


class FlightResultsPage:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.results_view = page.get_by_test_id("results-view")
        self.route = page.get_by_test_id("results-route")
        self.results = page.get_by_test_id("flight-results")
        self.continue_button = page.get_by_test_id("continue-to-fares")
        self.error = page.get_by_test_id("flight-error")

    def select_flight(self, flight_id: str) -> None:
        self.page.get_by_test_id(f"flight-option-{flight_id}").check()

    def continue_to_fares(self) -> None:
        self.continue_button.click()

    def error_text(self) -> str:
        return self.error.inner_text()

    def route_text(self) -> str:
        return self.route.inner_text()
