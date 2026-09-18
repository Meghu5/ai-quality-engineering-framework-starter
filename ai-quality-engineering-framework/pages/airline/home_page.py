from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Page


class HomePage:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.origin = page.get_by_test_id("origin-input")
        self.destination = page.get_by_test_id("destination-input")
        self.departure_date = page.get_by_test_id("departure-date-input")
        self.passenger_count = page.get_by_test_id("passenger-count-input")
        self.search_button = page.get_by_test_id("search-submit")
        self.error = page.get_by_test_id("search-error")
        self.search_view = page.get_by_test_id("search-view")

    def open(self, app_path: Path) -> None:
        self.page.goto(app_path.resolve().as_uri())

    def search(
        self,
        *,
        origin: str,
        destination: str,
        departure_date: str,
        passengers: int,
    ) -> None:
        self.origin.select_option(origin)
        self.destination.select_option(destination)
        self.departure_date.fill(departure_date)
        self.passenger_count.fill(str(passengers))
        self.search_button.click()

    def submit_empty_search(self) -> None:
        self.search_button.click()

    def error_text(self) -> str:
        return self.error.inner_text()
