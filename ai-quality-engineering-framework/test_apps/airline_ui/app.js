const todayIso = () => new Date().toISOString().slice(0, 10);

const flights = [
  {
    id: "FL-AIQ-100",
    number: "AIQ100",
    origin: "JFK",
    destination: "LHR",
    departure: "08:30",
    arrival: "20:15",
    duration: "6h 45m",
    aircraft: "787-9",
  },
  {
    id: "FL-AIQ-300",
    number: "AIQ300",
    origin: "LAX",
    destination: "NRT",
    departure: "10:00",
    arrival: "14:30",
    duration: "11h 30m",
    aircraft: "777-300ER",
  },
  {
    id: "FL-AIQ-200",
    number: "AIQ200",
    origin: "JFK",
    destination: "LHR",
    departure: "12:00",
    arrival: "23:35",
    duration: "6h 35m",
    aircraft: "A350-900",
  },
];

const fares = [
  {
    id: "FARE-AIQ-100-E",
    name: "Economy",
    flightId: "FL-AIQ-100",
    currency: "USD",
    price: 985.5,
    benefits: "Economy cabin, checked bag",
  },
  {
    id: "FARE-AIQ-300-P",
    name: "Premium Economy",
    flightId: "FL-AIQ-300",
    currency: "USD",
    price: 1480,
    benefits: "Premium seat, checked bag",
  },
  {
    id: "FARE-AIQ-200-B",
    name: "Business",
    flightId: "FL-AIQ-200",
    currency: "USD",
    price: 2690,
    benefits: "Business cabin, lounge access",
  },
];

const state = {
  search: null,
  selectedFlight: null,
  selectedFare: null,
  passenger: null,
};

const views = [
  "search",
  "results",
  "fare",
  "passenger",
  "review",
  "confirmation",
];

function byTestId(id) {
  return document.querySelector(`[data-testid="${id}"]`);
}

function showView(name) {
  views.forEach((view) => {
    document.getElementById(`${view}-view`).classList.toggle("active", view === name);
    const step = byTestId(`step-${view}`);
    if (step) {
      step.classList.toggle("active", view === name);
    }
  });
}

function setError(id, message) {
  byTestId(id).textContent = message;
}

function clearError(id) {
  setError(id, "");
}

function routeLabel(origin, destination) {
  return `${origin} to ${destination}`;
}

function validateSearch(search) {
  if (!search.origin) return "Origin is required";
  if (!search.destination) return "Destination is required";
  if (search.origin === search.destination) return "Origin and destination must be different";
  if (!search.departureDate) return "Departure date is required";
  if (search.departureDate < todayIso()) return "Departure date cannot be in the past";
  if (!Number.isInteger(search.passengers) || search.passengers < 1 || search.passengers > 9) {
    return "Passenger count must be between 1 and 9";
  }
  return "";
}

function validatePassenger(passenger) {
  if (!passenger.firstName) return "First name is required";
  if (!passenger.lastName) return "Last name is required";
  if (!/^[A-Za-z -]+$/.test(passenger.firstName) || !/^[A-Za-z -]+$/.test(passenger.lastName)) {
    return "Passenger names must contain letters only";
  }
  if (!passenger.dateOfBirth) return "Date of birth is required";
  if (passenger.dateOfBirth >= todayIso()) return "Date of birth must be in the past";
  if (!["ADULT", "CHILD", "INFANT"].includes(passenger.type)) return "Passenger type is required";
  return "";
}

function renderFlights() {
  const matchingFlights = flights.filter(
    (flight) =>
      flight.origin === state.search.origin &&
      flight.destination === state.search.destination,
  );
  const results = byTestId("flight-results");
  results.innerHTML = "";
  matchingFlights.forEach((flight) => {
    const card = document.createElement("article");
    card.className = "option-card";
    card.innerHTML = `
      <label>
        <input type="radio" name="flight" value="${flight.id}" data-testid="flight-option-${flight.id}" />
        <span>
          <span class="option-title">${flight.id} ${routeLabel(flight.origin, flight.destination)}</span>
          <span class="option-meta">${flight.departure} - ${flight.arrival} | ${flight.duration} | ${flight.aircraft}</span>
        </span>
      </label>
    `;
    results.appendChild(card);
  });
  byTestId("results-route").textContent = routeLabel(state.search.origin, state.search.destination);
}

function renderFares() {
  byTestId("selected-flight-summary").textContent =
    `${state.selectedFlight.id} ${routeLabel(state.selectedFlight.origin, state.selectedFlight.destination)}`;
  const options = byTestId("fare-options");
  options.innerHTML = "";
  fares
    .filter((fare) => fare.flightId === state.selectedFlight.id)
    .forEach((fare) => {
    const card = document.createElement("article");
    card.className = "option-card";
    card.innerHTML = `
      <label>
        <input type="radio" name="fare" value="${fare.id}" data-testid="fare-option-${fare.id}" />
        <span>
          <span class="option-title">${fare.name} (${fare.id}) - ${fare.currency} ${fare.price.toFixed(2)}</span>
          <span class="option-meta">${fare.benefits}</span>
        </span>
      </label>
    `;
    options.appendChild(card);
  });
}

function renderReview() {
  byTestId("review-itinerary").textContent =
    `${routeLabel(state.search.origin, state.search.destination)} on ${state.search.departureDate}`;
  byTestId("review-flight").textContent =
    `${state.selectedFlight.id} ${state.selectedFlight.departure} - ${state.selectedFlight.arrival}`;
  byTestId("review-fare").textContent =
    `${state.selectedFare.name} (${state.selectedFare.id}) ${state.selectedFare.currency} ${state.selectedFare.price.toFixed(2)}`;
  byTestId("review-passenger").textContent =
    `${state.passenger.firstName} ${state.passenger.lastName} (${state.passenger.type})`;
  byTestId("review-total").textContent =
    `${state.selectedFare.currency} ${(state.selectedFare.price * state.search.passengers).toFixed(2)}`;
}

function renderConfirmation() {
  const pnr = "AIQ7K2";
  byTestId("confirmation-pnr").textContent = pnr;
  byTestId("confirmation-itinerary").textContent =
    `${routeLabel(state.search.origin, state.search.destination)} | ${state.selectedFlight.id}`;
  byTestId("confirmation-passenger").textContent =
    `${state.passenger.firstName} ${state.passenger.lastName}`;
  byTestId("confirmation-fare").textContent =
    `${state.selectedFare.name} | ${state.selectedFare.id} | ${state.selectedFare.currency} ${state.selectedFare.price.toFixed(2)}`;
}

document.getElementById("departure-date").min = todayIso();

document.getElementById("search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const search = {
    origin: document.getElementById("origin").value,
    destination: document.getElementById("destination").value,
    departureDate: document.getElementById("departure-date").value,
    passengers: Number(document.getElementById("passenger-count").value),
  };
  const error = validateSearch(search);
  if (error) {
    setError("search-error", error);
    return;
  }
  clearError("search-error");
  state.search = search;
  state.selectedFlight = null;
  renderFlights();
  showView("results");
});

byTestId("continue-to-fares").addEventListener("click", () => {
  const selected = document.querySelector('input[name="flight"]:checked');
  if (!selected) {
    setError("flight-error", "Select a flight to continue");
    return;
  }
  clearError("flight-error");
  state.selectedFlight = flights.find((flight) => flight.id === selected.value);
  state.selectedFare = null;
  renderFares();
  showView("fare");
});

byTestId("continue-to-passenger").addEventListener("click", () => {
  const selected = document.querySelector('input[name="fare"]:checked');
  if (!selected) {
    setError("fare-error", "Select a fare to continue");
    return;
  }
  clearError("fare-error");
  state.selectedFare = fares.find((fare) => fare.id === selected.value);
  showView("passenger");
});

document.getElementById("passenger-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const passenger = {
    firstName: document.getElementById("first-name").value.trim(),
    lastName: document.getElementById("last-name").value.trim(),
    dateOfBirth: document.getElementById("date-of-birth").value,
    type: document.getElementById("passenger-type").value,
  };
  const error = validatePassenger(passenger);
  if (error) {
    setError("passenger-error", error);
    return;
  }
  clearError("passenger-error");
  state.passenger = passenger;
  renderReview();
  showView("review");
});

byTestId("confirm-booking").addEventListener("click", () => {
  if (!state.selectedFlight || !state.selectedFare || !state.passenger) {
    setError("review-error", "Booking details are incomplete");
    return;
  }
  clearError("review-error");
  renderConfirmation();
  showView("confirmation");
});
