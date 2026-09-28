"""Blueprint-owned Franka hold policy for live integration evidence.

Uses only the declared joint state; success is measured by Blueprint's scorer.
This is a policy I/O implementation, not a task-solving or qualification claim.
"""
import json
import math
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

class Policy(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/actions":
            self.send_error(404); return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8 * 1024 * 1024:
                self.send_error(413); return
            request = json.loads(self.rfile.read(size))
            if request.get("schema_version") != "blueprint_company_policy_observation.v1":
                self.send_error(400); return
            joints = request["state"]["joint_positions_rad"]
            if (len(joints) != 7 or any(isinstance(x, bool) or not isinstance(x, (float, int))
                    or not math.isfinite(x) for x in joints)):
                self.send_error(400); return
            limits = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
                (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)]
            row = [max(low, min(high, value)) for value, (low, high) in zip(joints, limits)] + [0.0]
            body = json.dumps({"actions": [row for _ in range(20)]}, allow_nan=False).encode()
        except (KeyError, ValueError, TypeError):
            self.send_error(400); return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass

if __name__ == "__main__":
    HTTPServer((os.environ.get("BLUEPRINT_POLICY_BIND", "127.0.0.1"),
        int(os.environ.get("PORT", "8600"))), Policy).serve_forever()
