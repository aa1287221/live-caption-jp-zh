"""Fake "exe" for llama_server.py tests: serves /health, controlled entirely via env vars.

Env vars (never argv, so ``ManagedLlamaServer.command`` stays exactly what production
builds and can be asserted on directly):
  FAKE_EXIT_CODE     if set, exit immediately with this code (never binds a port)
  FAKE_EXIT_MESSAGE  printed to stdout before exiting, for log-tail assertions
  FAKE_HEALTH_503_COUNT  number of /health polls to answer 503 (loading) before 200
"""

import argparse
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, required=True)
	parser.add_argument("-m", "--model", default=None)
	args, _extra = parser.parse_known_args()

	exit_code = os.environ.get("FAKE_EXIT_CODE")
	if exit_code is not None:
		message = os.environ.get("FAKE_EXIT_MESSAGE", "fake-llama-server: exiting early")
		print(message, flush=True)
		sys.exit(int(exit_code))

	pending = int(os.environ.get("FAKE_HEALTH_503_COUNT", "0"))

	class Handler(BaseHTTPRequestHandler):
		def do_GET(self):
			nonlocal pending
			if self.path != "/health":
				self.send_response(404)
				self.end_headers()
				return
			if pending > 0:
				pending -= 1
				body = b'{"error": {"code": 503, "message": "Loading model"}}'
				self.send_response(503)
			else:
				body = b'{"status": "ok"}'
				self.send_response(200)
			self.send_header("Content-Type", "application/json")
			self.send_header("Content-Length", str(len(body)))
			self.end_headers()
			self.wfile.write(body)

		def log_message(self, *_args):
			pass

	server = ThreadingHTTPServer((args.host, args.port), Handler)
	print(f"fake-llama-server: listening on {args.host}:{args.port}", flush=True)
	server.serve_forever()


if __name__ == "__main__":
	main()
