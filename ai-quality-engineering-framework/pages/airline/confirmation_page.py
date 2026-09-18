from __future__ import annotations

from playwright.sync_api import Page


class ConfirmationPage:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.confirmation_view = page.get_by_test_id("confirmation-view")
        self.message = page.get_by_test_id("confirmation-message")
        self.pnr = page.get_by_test_id("confirmation-pnr")
        self.itinerary = page.get_by_test_id("confirmation-itinerary")
        self.passenger = page.get_by_test_id("confirmation-passenger")
        self.fare = page.get_by_test_id("confirmation-fare")

    def pnr_text(self) -> str:
        return self.pnr.inner_text()
