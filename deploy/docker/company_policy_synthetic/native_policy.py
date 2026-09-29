"""Blueprint-owned hold-position controller for one development-only Franka task."""
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer

JOINT_LIMITS = (
    (-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
    (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)


class Policy(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/actions":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8 * 1024 * 1024:
                raise ValueError("size")
            wire = json.loads(self.rfile.read(length))
            state = wire["state"]
            joints = state["joint_positions_rad"]
            gripper = state["gripper_position"]
            if len(joints) != 7 or len(gripper) != 1:
                raise ValueError("shape")
            row = []
            for value, (lower, upper) in zip(joints, JOINT_LIMITS, strict=True):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("joint")
                row.append(max(lower, min(upper, float(value))))
            if isinstance(gripper[0], bool) or not isinstance(gripper[0], (int, float)) or not math.isfinite(gripper[0]):
                raise ValueError("gripper")
            row.append(max(0.0, min(1.0, float(gripper[0]))))
            body = json.dumps({"actions": [row for _ in range(20)]}, allow_nan=False).encode()
        except (ValueError, TypeError, KeyError, IndexError):
            self.send_error(400)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


HTTPServer(("127.0.0.1", 8600), Policy).serve_forever()
