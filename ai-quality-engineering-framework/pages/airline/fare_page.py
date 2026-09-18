from __future__ import annotations

from playwright.sync_api import Page


class FarePage:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.fare_view = page.get_by_test_id("fare-view")
        self.flight_summary = page.get_by_test_id("selected-flight-summary")
        self.fare_options = page.get_by_test_id("fare-options")
        self.continue_button = page.get_by_test_id("continue-to-passenger")
        self.error = page.get_by_test_id("fare-error")

    def select_fare(self, fare_id: str) -> None:
        self.page.get_by_test_id(f"fare-option-{fare_id}").check()

    def continue_to_passenger(self) -> None:
        self.continue_button.click()

    def error_text(self) -> str:
        return self.error.inner_text()

    def selected_flight_text(self) -> str:
        return self.flight_summary.inner_text()
