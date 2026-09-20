export const searchRequest = {
  origin: 'JFK',
  destination: 'LHR',
  departure_date: '2099-06-15',
  passengers: 1,
  cabin: 'ECONOMY',
  trip_type: 'ONE_WAY',
};

export const expectedFlight = {
  flight_id: 'FL-AIQ-100',
  origin: 'JFK',
  destination: 'LHR',
  fare_id: 'FARE-AIQ-100-E',
};

export const expectedFare = {
  fare_id: 'FARE-AIQ-100-E',
  flight_id: 'FL-AIQ-100',
  currency: 'USD',
  total_amount: '985.50',
};

export const passenger = {
  first_name: 'Avery',
  last_name: 'Stone',
  passenger_type: 'ADULT',
  date_of_birth: '1988-04-12',
};

export const expectedBooking = {
  booking_id: 'BKG-AIQ-0001',
  pnr: 'AIQ7K2',
  status: 'CONFIRMED',
};
