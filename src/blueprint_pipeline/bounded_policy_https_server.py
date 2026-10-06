"""Bound unauthenticated TLS and HTTP reads on authorized policy listeners."""
from __future__ import annotations

import io
import socketserver
import ssl
import sys
import threading
import time
from http.server import HTTPServer


HANDSHAKE_TIMEOUT_SECONDS = 5.0
REQUEST_READ_TIMEOUT_SECONDS = 10.0
MAXIMUM_CONNECTIONS = 8


class _DeadlineReader(io.RawIOBase):
    def __init__(self, connection, seconds):
        self.connection = connection
        self.deadline = time.monotonic() + seconds

    def readable(self):
        return True

    def readinto(self, buffer):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("policy_https_request_read_deadline")
        self.connection.settimeout(remaining)
        return self.connection.recv_into(buffer)


def bound_policy_request_reads(handler):
    """One deadline for headers and body, including peers that drip bytes."""
    handler.connection.settimeout(REQUEST_READ_TIMEOUT_SECONDS)
    handler.rfile.close()
    reader = _DeadlineReader(handler.connection, REQUEST_READ_TIMEOUT_SECONDS)
    handler.rfile = io.BufferedReader(reader)


class BoundedPolicyHTTPSServer(socketserver.ThreadingMixIn, HTTPServer):
    # Admission counts both TLS handshakes and handlers. A stalled handshake
    # cannot block accept or create an unbounded population of reader threads.
    daemon_threads = True
    block_on_close = False

    def __init__(self, address, handler, *, context):
        self.context = context
        self._slots = threading.BoundedSemaphore(MAXIMUM_CONNECTIONS)
        self._connections = set()
        self._connections_lock = threading.Lock()
        self._closing = False
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        with self._connections_lock:
            if self._closing or not self._slots.acquire(blocking=False):
                self.shutdown_request(request)
                return
            self._connections.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            with self._connections_lock:
                self._connections.discard(request)
            self.shutdown_request(request)
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        secured = None
        try:
            request.settimeout(HANDSHAKE_TIMEOUT_SECONDS)
            secured = self.context.wrap_socket(request, server_side=True,
                do_handshake_on_connect=False)
            with self._connections_lock:
                self._connections.discard(request)
                self._connections.add(secured)
                if self._closing:
                    return
            try:
                secured.do_handshake()
            except (OSError, ssl.SSLError):
                return  # Unauthenticated peers cannot produce private logs.
            secured.settimeout(REQUEST_READ_TIMEOUT_SECONDS)
            super().process_request_thread(secured, client_address)
        except (OSError, ssl.SSLError):
            return
        finally:
            with self._connections_lock:
                self._connections.discard(request)
                self._connections.discard(secured)
            self.shutdown_request(secured if secured is not None else request)
            self._slots.release()

    def server_close(self):
        with self._connections_lock:
            self._closing = True
            connections = tuple(self._connections)
        for connection in connections:
            self.shutdown_request(connection)
        super().server_close()

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], OSError):
            return  # Expected timeouts/disconnects carry no private diagnostics.
        super().handle_error(request, client_address)
