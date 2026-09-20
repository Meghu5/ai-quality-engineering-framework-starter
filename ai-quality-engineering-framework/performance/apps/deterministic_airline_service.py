from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from utils.airline.deterministic_api import (
    DeterministicAirlineBackend,
    load_airline_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "airline"


class AirlinePerformanceHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    backend: DeterministicAirlineBackend

    def handle(self) -> None:
        try:
            super().handle()
        except ConnectionResetError:
            return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, {"status": "ok", "service": "airline-performance"})
            return

        status_code, body = self.backend.handle(method="GET", path=path)
        self._send_json(status_code, body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_json_body()
        except ValueError as exc:
            self._send_json(422, {"error": str(exc)})
            return

        status_code, response_body = self.backend.handle(
            method="POST",
            path=path,
            body=body,
        )
        self._send_json(status_code, response_body)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        if not raw_body:
            return {}

        try:
            body = json.loads(raw_body.decode("utf-8"))
        except ValueError as exc:
            raise ValueError("malformed JSON") from exc

        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object")
        return body

    def _send_json(self, status_code: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def create_server(host: str, port: int) -> ThreadingHTTPServer:
    flights = load_airline_json(DATA_DIR, "flights.json")
    fares = load_airline_json(DATA_DIR, "fares.json")
    AirlinePerformanceHandler.backend = DeterministicAirlineBackend(flights, fares)
    return ThreadingHTTPServer((host, port), AirlinePerformanceHandler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic airline HTTP test double for k6 performance "
            "testing. This is not a production airline backend."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    server = create_server(args.host, args.port)
    print(f"Airline performance service listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
