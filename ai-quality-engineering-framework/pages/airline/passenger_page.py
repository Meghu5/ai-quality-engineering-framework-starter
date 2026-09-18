from __future__ import annotations

from playwright.sync_api import Page


class PassengerPage:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.passenger_view = page.get_by_test_id("passenger-view")
        self.first_name = page.get_by_test_id("first-name-input")
        self.last_name = page.get_by_test_id("last-name-input")
        self.date_of_birth = page.get_by_test_id("date-of-birth-input")
        self.passenger_type = page.get_by_test_id("passenger-type-input")
        self.continue_button = page.get_by_test_id("continue-to-review")
        self.error = page.get_by_test_id("passenger-error")

    def enter_passenger(
        self,
        *,
        first_name: str,
        last_name: str,
        date_of_birth: str,
        passenger_type: str,
    ) -> None:
        self.first_name.fill(first_name)
        self.last_name.fill(last_name)
        self.date_of_birth.fill(date_of_birth)
        self.passenger_type.select_option(passenger_type)

    def continue_to_review(self) -> None:
        self.continue_button.click()

    def error_text(self) -> str:
        return self.error.inner_text()
