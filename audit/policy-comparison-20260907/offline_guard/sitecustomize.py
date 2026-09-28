"""Audit-only network tripwire, inherited by Python child interpreters."""
import socket
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex

def _connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError('policy_integrity_audit_external_edge_forbidden')
    return _original_connect(self, address)

def _connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError('policy_integrity_audit_external_edge_forbidden')
    return _original_connect_ex(self, address)

socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex
