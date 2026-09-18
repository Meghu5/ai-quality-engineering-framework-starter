from __future__ import annotations

from playwright.sync_api import Page


class ReviewBookingPage:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.review_view = page.get_by_test_id("review-view")
        self.itinerary = page.get_by_test_id("review-itinerary")
        self.flight = page.get_by_test_id("review-flight")
        self.fare = page.get_by_test_id("review-fare")
        self.passenger = page.get_by_test_id("review-passenger")
        self.total = page.get_by_test_id("review-total")
        self.confirm_button = page.get_by_test_id("confirm-booking")
        self.error = page.get_by_test_id("review-error")

    def confirm_booking(self) -> None:
        self.confirm_button.click()
