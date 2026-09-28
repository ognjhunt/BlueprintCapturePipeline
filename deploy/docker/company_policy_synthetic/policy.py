"""Blueprint-owned zero-action policy. Synthetic interface proof only."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class Policy(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/actions":
            self.send_error(404)
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if request.get("synthetic") is not True:
            self.send_error(403)
            return
        body = json.dumps({"actions": [[0.0, 0.5] for _ in range(15)]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass

HTTPServer(("127.0.0.1", 8600), Policy).serve_forever()
