import http from 'k6/http';

import { currentEnvironment } from '../config/environments.js';

const defaultHeaders = {
  Accept: 'application/json',
  'Content-Type': 'application/json',
};

export function baseUrl() {
  return currentEnvironment().baseUrl;
}

export function get(path, tags = {}) {
  return http.get(`${baseUrl()}${path}`, {
    headers: defaultHeaders,
    tags,
  });
}

export function post(path, body, tags = {}) {
  return http.post(`${baseUrl()}${path}`, JSON.stringify(body), {
    headers: defaultHeaders,
    tags,
  });
}

export function json(response) {
  try {
    return response.json();
  } catch (error) {
    return {};
  }
}
