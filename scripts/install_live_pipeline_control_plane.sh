#!/usr/bin/env bash
set -euo pipefail

# Resolve a promoted-release symlink before ownership changes.  Without the
# physical path, recursive chown changes the link rather than the detached
# checkout Git resolves, leaving the service account unable to prove identity.
REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
ENV_DIR="${ENV_DIR:-/etc/blueprint}"
ENV_FILE="${ENV_FILE:-${ENV_DIR}/pipeline-control-plane.env}"
STATE_DIR="${STATE_DIR:-/var/lib/blueprint/pipeline-control-plane}"
HANDOFF_DIR="${HANDOFF_DIR:-/var/lib/blueprint/pubsub-handoffs}"
PROVIDER_SECRETS_DIR="${PROVIDER_SECRETS_DIR:-${ENV_DIR}/provider-secrets}"
CREDENTIALS_DIR="${CREDENTIALS_DIR:-${ENV_DIR}/credentials}"
LAUNCH_PROFILE_DIR="${LAUNCH_PROFILE_DIR:-${ENV_DIR}/task-evaluation-launch-profiles}"
CONFIGURED_CONTROLS_PLAN_ROOT="${CONFIGURED_CONTROLS_PLAN_ROOT:-${ENV_DIR}/task-evaluation-configured-controls-plans}"
CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT="${CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT:-${ENV_DIR}/task-evaluation-configured-controls-intents}"
CONFIGURED_CONTROLS_WEBAPP_SECRET="${CONFIGURED_CONTROLS_WEBAPP_SECRET:-${PROVIDER_SECRETS_DIR}/blueprint_task_evaluation_launch_submit_secret}"
TASK_EVALUATION_INPUT_ROOT="${TASK_EVALUATION_INPUT_ROOT:-/var/lib/blueprint/task-evaluation-inputs}"
WORK_VOLUME_ROOT="${WORK_VOLUME_ROOT:-/mnt/blueprint-work}"
CAPTURE_RECONSTRUCTION_POLICY_DIR="${CAPTURE_RECONSTRUCTION_POLICY_DIR:-${ENV_DIR}/capture-reconstruction-policies}"
CADDY_SITE_FILE="${CADDY_SITE_FILE:-/etc/caddy/Caddyfile}"
SERVICE_USER="${SERVICE_USER:-blueprint}"
SERVICE_GROUP="${SERVICE_GROUP:-blueprint}"
ENABLE_NOW=false
DRY_RUN=false

usage() {
  cat <<'USAGE'
Usage: scripts/install_live_pipeline_control_plane.sh [--enable-now] [--dry-run]

Installs the Blueprint live pipeline control-plane systemd service/timer, the
capture handoff Pub/Sub listener service/timer, and the optional authenticated
WebApp intake service unit, plus the paid-work spend admission guard service/
timer and read-only provider billing reconciler. The spend guard remains locked
until current provider billing input and credentials are installed.
The service runs one safe control-plane pass on each timer tick:
read env, audit readiness, optionally consume the robot-eval job inbox, write
manifests, run the proof-boundary audit, and exit. It does not add secrets or
enable live simulator/provider actions by itself.

Environment overrides:
  SYSTEMD_DIR=/etc/systemd/system
  ENV_DIR=/etc/blueprint
  ENV_FILE=/etc/blueprint/pipeline-control-plane.env
  STATE_DIR=/var/lib/blueprint/pipeline-control-plane
  HANDOFF_DIR=/var/lib/blueprint/pubsub-handoffs
  PROVIDER_SECRETS_DIR=/etc/blueprint/provider-secrets
  CREDENTIALS_DIR=/etc/blueprint/credentials
  LAUNCH_PROFILE_DIR=/etc/blueprint/task-evaluation-launch-profiles
  TASK_EVALUATION_INPUT_ROOT=/var/lib/blueprint/task-evaluation-inputs
  WORK_VOLUME_ROOT=/mnt/blueprint-work
  CAPTURE_RECONSTRUCTION_POLICY_DIR=/etc/blueprint/capture-reconstruction-policies
  CADDY_SITE_FILE=/etc/caddy/Caddyfile
  SERVICE_USER=blueprint
  SERVICE_GROUP=blueprint

The public TLS/reverse-proxy edge is rendered from deploy/caddy/Caddyfile only
when BLUEPRINT_PIPELINE_PUBLIC_HOSTNAME is set, for example:

  BLUEPRINT_PIPELINE_PUBLIC_HOSTNAME=pipeline.example.com \
    scripts/install_live_pipeline_control_plane.sh
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable-now)
      ENABLE_NOW=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run] %q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ "${EUID}" -ne 0 && "${DRY_RUN}" != "true" ]]; then
  echo "Run as root or use --dry-run." >&2
  exit 1
fi
if [[ -L "${WORK_VOLUME_ROOT}" ]]; then
  echo "ERROR: work volume root is a symlink" >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "true" ]] && ! mountpoint -q -- "${WORK_VOLUME_ROOT}"; then
  echo "ERROR: mount the work volume before installing the lane scratch root" >&2
  exit 1
fi

if ! getent group "${SERVICE_GROUP}" >/dev/null 2>&1; then
  run groupadd --system "${SERVICE_GROUP}"
fi
if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
  run useradd --system --gid "${SERVICE_GROUP}" --home-dir /nonexistent \
    --shell /usr/sbin/nologin "${SERVICE_USER}"
fi

# Prepare immutable system-ABI runtime before changing source ownership or
# exposing units. Admit the exact main CI manifest with the protected host gh
# before publishing or executing either candidate installer or verifier module.
# This embedded bootstrap is tested against the normal deployer's functions;
# importing the mutable checkout deployer here would bypass that first proof.
if [[ "${DRY_RUN}" == "true" ]]; then
  run /usr/bin/python3 -I -S "${REPO_ROOT}/scripts/install_scene_retirement_runtime.py" \
    --source "${REPO_ROOT}" --source-commit "<validated-Git-commit>" --locked-sdk \
    --source-manifest "<protected-and-verified-source-manifest>" \
    --source-attestation "<protected-source-provenance-bundle>" \
    --manifest-verifier "<SHA256-admitted-verifier-module>"
else
  /usr/bin/python3 -I -S - "${REPO_ROOT}" <<'PY_RUNTIME'
from __future__ import annotations
import functools, http.client, selectors, socket, urllib.error
import base64, fcntl, hashlib, json, math, os, pathlib, re, signal, stat, subprocess, sys, tempfile, time, urllib.parse, urllib.request
Path = pathlib.Path
ControlPlaneDeployError = ValueError
_SCENE_RUNTIME_BOOT_ROOT = Path("/usr/lib/blueprint/scene-retirement-runtime")
_SCENE_RUNTIME_OWNER = 0
_SCENE_RUNTIME_INSTALL_SECONDS = 900
# The service-owned state directory cannot be an ancestor of root-admitted proof.
_SCENE_SOURCE_ATTESTATIONS = _SCENE_RUNTIME_BOOT_ROOT / "source-attestations"
_SCENE_SOURCE_GH = Path("/usr/bin/gh")

_SOURCE_MAX_RESOLVER_BYTES = 64 * 1024
_SOURCE_MAX_RESOLVER_ADDRESSES = 256


_SOURCE_RESOLVER_SCRIPT = """
import json, socket, sys
host, port = json.loads(sys.stdin.buffer.read().decode("utf-8"))
try:
    addresses = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    result = ({"addresses": addresses} if len(addresses) <= 256 else
              {"refusal": "controlled_http_resolution_oversized"})
except socket.gaierror as error:
    result = {"error": [error.errno, error.strerror]}
sys.stdout.buffer.write(json.dumps(result).encode("utf-8"))
"""


def _source_connection_remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("controlled_http_response_timeout")
    return remaining


def _source_resolve_addresses(address, *, deadline):
    """Bound libc resolution without accumulating uncancellable threads.

    Socket timeouts do not cover getaddrinfo. The isolated child receives only
    the hostname and port, and is killed and reaped on timeout or interruption.
    It never receives the URL, headers, credentials, or request body.
    """
    _source_connection_remaining(deadline)
    payload = json.dumps(address).encode("utf-8")
    if len(payload) > 4096:
        raise ValueError("controlled_http_resolution_oversized")
    _source_connection_remaining(deadline)
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", _SOURCE_RESOLVER_SCRIPT],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env={},
    )
    try:
        raw = bytearray()
        written = 0
        with selectors.DefaultSelector() as selector:
            os.set_blocking(process.stdin.fileno(), False)
            os.set_blocking(process.stdout.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE)
            selector.register(process.stdout, selectors.EVENT_READ)
            while not process.stdout.closed:
                events = selector.select(_source_connection_remaining(deadline))
                if not events:
                    raise TimeoutError("controlled_http_response_timeout")
                for key, _event in events:
                    if key.fileobj is process.stdin:
                        written += os.write(process.stdin.fileno(), payload[written:written + 512])
                        if written == len(payload):
                            selector.unregister(process.stdin)
                            process.stdin.close()
                    else:
                        chunk = os.read(process.stdout.fileno(),
                                        min(8192, _SOURCE_MAX_RESOLVER_BYTES + 1 - len(raw)))
                        if not chunk:
                            selector.unregister(process.stdout)
                            process.stdout.close()
                        else:
                            raw.extend(chunk)
                            if len(raw) > _SOURCE_MAX_RESOLVER_BYTES:
                                raise ValueError("controlled_http_resolution_oversized")
                _source_connection_remaining(deadline)
        try:
            process.wait(timeout=_source_connection_remaining(deadline))
        except subprocess.TimeoutExpired:
            raise TimeoutError("controlled_http_response_timeout") from None
        _source_connection_remaining(deadline)
        if process.returncode != 0:
            raise OSError("controlled_http_resolution_failed")
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise ValueError("controlled_http_resolution_invalid") from None
        _source_connection_remaining(deadline)
        addresses = _source_validated_resolver_result(result)
        _source_connection_remaining(deadline)
        return addresses
    finally:
        if process.poll() is None:
            process.kill()
        process.stdin.close()
        process.stdout.close()
        process.wait()


def _source_validated_resolver_result(result):
    if type(result) is not dict:
        raise ValueError("controlled_http_resolution_invalid")
    if set(result) == {"refusal"} and result["refusal"] == "controlled_http_resolution_oversized":
        raise ValueError("controlled_http_resolution_oversized")
    if set(result) == {"error"}:
        error = result["error"]
        if (type(error) is list and len(error) == 2
                and type(error[0]) is int and type(error[1]) is str):
            raise socket.gaierror(*error)
        raise ValueError("controlled_http_resolution_invalid")
    rows = result.get("addresses")
    if set(result) != {"addresses"} or type(rows) is not list:
        raise ValueError("controlled_http_resolution_invalid")
    if len(rows) > _SOURCE_MAX_RESOLVER_ADDRESSES:
        raise ValueError("controlled_http_resolution_oversized")
    addresses = []
    for row in rows:
        if type(row) is not list or len(row) != 5:
            raise ValueError("controlled_http_resolution_invalid")
        family, kind, protocol, canonical, sockaddr = row
        if (type(family) is not int or family not in {socket.AF_INET, socket.AF_INET6}
                or type(kind) is not int or kind != socket.SOCK_STREAM
                or type(protocol) is not int or protocol not in {0, socket.IPPROTO_TCP}
                or type(canonical) is not str or type(sockaddr) is not list
                or len(sockaddr) != (2 if family == socket.AF_INET else 4)
                or type(sockaddr[0]) is not str or type(sockaddr[1]) is not int
                or not 0 <= sockaddr[1] <= 65535
                or any(type(value) is not int or not 0 <= value <= 0xffffffff
                       for value in sockaddr[2:])):
            raise ValueError("controlled_http_resolution_invalid")
        try:
            socket.inet_pton(family, sockaddr[0])
        except (OSError, ValueError):
            raise ValueError("controlled_http_resolution_invalid") from None
        addresses.append((family, kind, protocol, canonical, tuple(sockaddr)))
    return addresses


def _source_create_deadline_connection(address, _timeout=None, source_address=None, *, deadline):
    addresses = _source_resolve_addresses(address, deadline=deadline)
    sources = (_source_resolve_addresses(source_address, deadline=deadline)
               if source_address else None)
    last_error = None
    for family, kind, protocol, _canonical, sockaddr in addresses:
        _source_connection_remaining(deadline)
        peer = socket.socket(family, kind, protocol)
        try:
            peer.settimeout(_source_connection_remaining(deadline))
            if sources:
                source = next((item[4] for item in sources if item[0] == family), None)
                if source is None:
                    raise OSError("controlled_http_source_address_unavailable")
                peer.bind(source)
            peer.connect(sockaddr)
            _source_connection_remaining(deadline)
            # CONNECT status/headers must share the budget before TLS starts.
            return _SourceDeadlineSocket(peer, deadline, 30)
        except OSError as error:
            last_error = error
            peer.close()
        except BaseException:
            peer.close()
            raise
    _source_connection_remaining(deadline)
    if last_error is not None:
        raise last_error
    raise OSError("getaddrinfo returns an empty list")


class _SourceDeadlineSocket:
    """Clamp every raw receive, including status/header/chunk line refills."""

    def __init__(self, socket, deadline: float, maximum_timeout: float):
        self._socket, self._deadline, self._maximum_timeout = socket, deadline, maximum_timeout

    def __getattr__(self, name):
        return getattr(self._socket, name)

    def _remaining(self):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("controlled_http_response_timeout")
        return min(remaining,self._maximum_timeout)

    def recv_into(self, target, *args):
        self._socket.settimeout(self._remaining())
        result = self._socket.recv_into(target, *args)
        self._remaining()
        return result

    def sendall(self, data, *args):
        self._socket.settimeout(self._remaining())
        self._socket.sendall(data, *args)
        self._remaining()

    def makefile(self, *args, **kwargs):
        stream = self._socket.makefile(*args, **kwargs)
        raw = getattr(stream, "raw", None)
        if raw is None or getattr(raw, "_sock", None) is not self._socket:
            stream.close()
            raise ValueError("controlled_http_response_reader_invalid")
        raw._sock = self
        return stream

class _SourceDeadlineHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, deadline, **kwargs):
        self._deadline = deadline
        super().__init__(*args, **kwargs)

    def connect(self):
        # The setup phases share one <=30-second budget, itself clamped to
        # the original installation deadline. Body reads retain global900
        # and the original per-read timeout after the verified TLS handshake.
        if type(self.timeout) not in {int,float} or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('controlled_http_connection_timeout_invalid')
        maximum_timeout = min(30,self.timeout)
        setup_deadline = min(self._deadline,time.monotonic()+maximum_timeout)
        _source_connection_remaining(setup_deadline)
        self._create_connection = functools.partial(_source_create_deadline_connection,deadline=setup_deadline)
        try:
            http.client.HTTPConnection.connect(self)
            peer = self.sock._socket
            peer.settimeout(_source_connection_remaining(setup_deadline))
            hostname = self._tunnel_host or self.host
            self.sock = self._context.wrap_socket(peer,server_hostname=hostname,do_handshake_on_connect=False)
            self.sock.settimeout(_source_connection_remaining(setup_deadline))
            self.sock.do_handshake()
            _source_connection_remaining(setup_deadline)
            _source_connection_remaining(self._deadline)
            self.sock = _SourceDeadlineSocket(self.sock,self._deadline,maximum_timeout)
        except BaseException:
            self.close()
            raise

class _SourceDeadlineHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, deadline, context):
        super().__init__(context=context)
        self.deadline = deadline

    def https_open(self, request):
        return self.do_open(lambda *args, **kwargs: _SourceDeadlineHTTPSConnection(
            *args, deadline=self.deadline, **kwargs), request, context=self._context)

class _SourceClosingHTTPErrorProcessor(urllib.request.HTTPErrorProcessor):
    def http_response(self, request, response):
        try:
            return super().http_response(request, response)
        except urllib.error.HTTPError as error:
            error.close()
            raise

    https_response = http_response

class _SourceNoRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self,error):
        super().__init__()
        self.error = error

    def redirect_request(self,request,response,code,message,headers,url):
        raise ValueError(self.error)

    def http_error_302(self,request,response,code,message,headers):
        try:
            raise ValueError(self.error)
        finally:
            response.close()

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _source_response_opener(deadline,*,error):
    if type(deadline) not in {int,float} or not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise ValueError(error)
    return urllib.request.build_opener(urllib.request.ProxyHandler({}),
        _SourceNoRedirect(error),_SourceDeadlineHTTPSHandler(deadline,None),_SourceClosingHTTPErrorProcessor())


def _scene_source_snappy(raw: bytes, *, deadline: float, cap: int) -> bytes:
    """Bound GitHub's raw Snappy bundle encoding; decoded bytes stay untrusted.

    Format: https://github.com/google/snappy/blob/main/format_description.txt
    No framed streams, allocation from unchecked lengths, or external codec.
    """
    error = "deploy_scene_retirement_runtime_unproven"
    def require(value):
        if not value:
            raise ControlPlaneDeployError(error)
    require(type(raw) is bytes and 0 < len(raw) <= 32*1024*1024 and 0 < cap <= 32*1024*1024)
    declared, cursor = 0, 0
    for shift in range(0,35,7):
        require(cursor < len(raw) and time.monotonic() <= deadline)
        value = raw[cursor]
        cursor += 1
        declared |= (value & 127) << shift
        if value < 128:
            break
    else:
        raise ControlPlaneDeployError(error)
    require(0 < declared <= cap)
    output = bytearray()
    while cursor < len(raw):
        require(time.monotonic() <= deadline)
        tag = raw[cursor]
        cursor += 1
        kind = tag & 3
        if kind == 0:
            length = tag >> 2
            if length >= 60:
                count = length-59
                require(cursor+count <= len(raw))
                length = int.from_bytes(raw[cursor:cursor+count],'little')
                cursor += count
            length += 1
            require(len(output)+length <= declared and cursor+length <= len(raw))
            output.extend(raw[cursor:cursor+length])
            cursor += length
        else:
            count = 1 if kind == 1 else 2 if kind == 2 else 4
            require(cursor+count <= len(raw))
            offset = int.from_bytes(raw[cursor:cursor+count],'little')
            cursor += count
            if kind == 1:
                offset |= (tag & 224) << 3
                length = 4+((tag >> 2) & 7)
            else:
                length = 1+(tag >> 2)
            require(0 < offset <= len(output) and len(output)+length <= declared)
            start = len(output)-offset
            # Snappy copies may overlap. Read only a <=64-byte seed instead
            # of duplicating the entire output for a large back-reference.
            seed = bytes(output[start:start+min(length,offset)])
            output.extend((seed*((length+len(seed)-1)//len(seed)))[:length])
    require(len(output) == declared and time.monotonic() <= deadline)
    return bytes(output)


def _scene_source_selector_bytes(source_commit: str) -> bytes:
    """Derive public lookup data without executing any candidate source."""
    if type(source_commit) is not str or re.fullmatch('[0-9a-f]{40}', source_commit) is None:
        raise ControlPlaneDeployError("deploy_scene_retirement_runtime_unproven")
    return (json.dumps({'schema_version':'blueprint.source_commit_selector.v1',
        'repository':'ognjhunt/BlueprintCapturePipeline', 'source_commit':source_commit},
        sort_keys=True, separators=(',', ':'), ensure_ascii=False)+'\n').encode()


def _scene_source_delivery(source_commit: str, *, deadline: float) -> None:
    """Acquire public proof as untrusted data; this function grants no admission."""
    error = "deploy_scene_retirement_runtime_unproven"
    repository = 'ognjhunt/BlueprintCapturePipeline'
    predicate_type = 'https://github.com/ognjhunt/BlueprintCapturePipeline/attestations/source-sha256-manifest/v1'
    def require(value):
        if not value:
            raise ControlPlaneDeployError(error)
    require(re.fullmatch('[0-9a-f]{40}', source_commit) is not None)
    opener = _source_response_opener(deadline,error=error)
    transfer_remaining = [48*1024*1024]
    decoded_remaining = [48*1024*1024]
    def fetch(url, cap, *, blob=False):
        require(time.monotonic() <= deadline)
        cap = min(cap, transfer_remaining[0])
        require(cap > 0)
        request = urllib.request.Request(url, headers={'Accept':'application/vnd.github+json',
            'X-GitHub-Api-Version':'2022-11-28', 'User-Agent':'Blueprint-source-admission'})
        with opener.open(request, timeout=min(30, max(.001, deadline-time.monotonic()))) as response:
            require(response.status == 200 and response.url == url)
            body = bytearray()
            while True:
                require(time.monotonic() <= deadline)
                # Each blocking read shares the absolute deadline, including a
                # slow peer that repeatedly delivers a tiny successful prefix.
                if response.fp is None:
                    break
                response.fp.raw._sock.settimeout(min(30, max(.001, deadline-time.monotonic())))
                chunk = response.read1(min(65536, cap+1-len(body)))
                if not chunk:
                    break
                require(len(body)+len(chunk) <= cap)
                body.extend(chunk)
                transfer_remaining[0] -= len(chunk)
            require(time.monotonic() <= deadline)
            raw = bytes(body)
            if blob:
                content_type = response.headers.get('Content-Type','').split(';',1)[0].lower()
                if content_type == 'application/x-snappy':
                    raw = _scene_source_snappy(raw,deadline=deadline,cap=min(32*1024*1024,decoded_remaining[0]))
                else:
                    require(content_type in {'application/json','application/vnd.dev.sigstore.bundle.v0.3+json'})
            require(len(raw) <= decoded_remaining[0])
            decoded_remaining[0] -= len(raw)
            return json.loads(raw)
    # GitHub's digest-addressed attestation API retains signed statements
    # independently of Actions artifacts. The known selector is a second
    # subject of the manifest attestation; it supplies no source authority.
    selector_digest = hashlib.sha256(_scene_source_selector_bytes(source_commit)).hexdigest()
    evidence = fetch('https://api.github.com/repos/'+repository+'/attestations/sha256:'+selector_digest+'?per_page=20', 48*1024*1024)
    require(type(evidence) is dict and type(evidence.get('attestations')) is list and 0 < len(evidence['attestations']) <= 20)
    candidates = []
    for item in evidence['attestations']:
        if type(item) is not dict:
            continue
        bundle = item.get('bundle')
        if type(bundle) is not dict:
            # The documented API can omit inline bytes and return a fresh
            # Azure blob URL. Restrict its origin/path before any request;
            # never follow redirects or forward authorization/proxy state.
            url = item.get('bundle_url')
            if type(url) is not str or len(url) > 16384 or any(ord(char) < 33 or ord(char) == 127 for char in url):
                continue
            try:
                location = urllib.parse.urlsplit(url)
            except ValueError:
                continue
            if (location.scheme != 'https' or location.netloc != 'tmaproduction.blob.core.windows.net'
                    or location.fragment or not re.fullmatch(
                        r'/attestations/[0-9]+/[0-9]{4}/[0-9]{2}/[0-9]{2}/[0-9]+\.json\.sn',location.path)):
                continue
            try:
                bundle = fetch(url,32*1024*1024,blob=True)
            except (OSError, ValueError):
                # Do not expose temporary signed query strings in errors.
                require(time.monotonic() <= deadline)
                continue
            if type(bundle) is not dict:
                continue
        envelope = bundle.get('dsseEnvelope')
        if (type(envelope) is not dict or envelope.get('payloadType') != 'application/vnd.in-toto+json'
                or type(envelope.get('payload')) is not str):
            continue
        require(len(envelope['payload']) <= 32*1024*1024)
        payload = base64.b64decode(envelope['payload'], validate=True)
        require(len(payload) <= 24*1024*1024)
        statement = json.loads(payload)
        if (type(statement) is not dict or statement.get('_type') != 'https://in-toto.io/Statement/v1'
                or statement.get('predicateType') != predicate_type or type(statement.get('predicate')) is not dict
                or type(statement.get('subject')) is not list or len(statement['subject']) != 2
                or not all(type(subject) is dict for subject in statement['subject'])):
            continue
        raw = (json.dumps(statement['predicate'], sort_keys=True, separators=(',', ':'), ensure_ascii=False)+'\n').encode()
        require(0 < len(raw) <= 16*1024*1024)
        digest = hashlib.sha256(raw).hexdigest()
        if (sum(subject.get('digest') == {'sha256':digest} for subject in statement['subject']) != 1
                or sum(subject.get('digest') == {'sha256':selector_digest} for subject in statement['subject']) != 1):
            continue
        proof = (json.dumps(bundle, sort_keys=True, separators=(',', ':'))+'\n').encode()
        require(len(proof) <= 32*1024*1024)
        candidates.append((raw, proof))
    require(candidates)
    root = _SCENE_SOURCE_ATTESTATIONS / source_commit
    if all((root/name).exists() for name in ('source-sha256-manifest.json','source-provenance.sigstore.json')):
        _scene_source_attestation(source_commit,deadline=deadline,_proof_root=root)
        return
    selection = _scene_source_selected_proof(root, deadline=deadline)
    selected = None
    # CI reruns at one SHA legitimately produce several signatures over the
    # same deterministic subject. List order is only a bounded trial order;
    # no candidate is admitted until the protected verifier checks its policy.
    for raw, proof in candidates:
        require(time.monotonic() <= deadline)
        if selection is not None and not (hashlib.sha256(proof).hexdigest()+'\n').encode().startswith(selection):
            continue
        candidate = root/'candidates'/hashlib.sha256(proof).hexdigest()
        _scene_source_cache_publish(candidate, raw, proof, deadline=deadline)
        try:
            _scene_source_attestation(source_commit, deadline=deadline, _proof_root=candidate)
        except (ControlPlaneDeployError, OSError, ValueError, subprocess.SubprocessError):
            require(time.monotonic() <= deadline)
            continue
        selected = (raw,proof)
        break
    require(selected is not None)
    _scene_source_cache_publish(root, *selected, deadline=deadline)
def _scene_source_selected_proof(root: Path, *, deadline: float) -> bytes | None:
    """Retain a protected selection across interrupted publication and CI reruns."""
    error = "deploy_scene_retirement_runtime_unproven"
    def require(value):
        if not value:
            raise ControlPlaneDeployError(error)
    require(root.is_absolute() and '..' not in root.parts and time.monotonic() <= deadline)
    held, identities = [], []
    def identity(info):
        return (info.st_dev,info.st_ino,info.st_mode,info.st_uid,info.st_gid,
                info.st_nlink,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
    try:
        directory = os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
        held.append(directory)
        for part in root.parts[1:]:
            before = os.fstat(directory)
            require(stat.S_ISDIR(before.st_mode) and before.st_uid in {0,_SCENE_RUNTIME_OWNER}
                    and not before.st_mode & 0o022)
            identities.append((directory,before))
            try:
                directory = os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=directory)
            except FileNotFoundError:
                return None
            held.append(directory)
        before = os.fstat(directory)
        require(stat.S_ISDIR(before.st_mode) and before.st_uid in {0,_SCENE_RUNTIME_OWNER} and not before.st_mode & 0o022)
        identities.append((directory,before))
        for name, complete in (('.public-selected-proof.sha256',True),('.public-selected-proof.sha256.public-pending',False)):
            try:
                fd = os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=directory)
            except FileNotFoundError:
                continue
            held.append(fd)
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and before.st_uid == _SCENE_RUNTIME_OWNER and before.st_nlink == 1
                    and stat.S_IMODE(before.st_mode) in {0o600,0o444} and 0 <= before.st_size <= 65
                    and (not complete or before.st_size == 65))
            raw = os.read(fd,66)
            require(len(raw) == before.st_size and re.fullmatch(rb'[0-9a-f]{0,64}\n?',raw) is not None
                    and (b'\n' not in raw or len(raw) == 65)
                    and identity(before) == identity(os.fstat(fd)) == identity(os.stat(name,dir_fd=directory,follow_symlinks=False)))
            for retained, original in identities:
                require(identity(os.fstat(retained)) == identity(original))
            return raw
        return None
    finally:
        for fd in reversed(held):
            os.close(fd)

def _scene_source_cache_publish(root: Path, raw: bytes, proof: bytes, *, deadline: float) -> None:
    """Publish bounded proof DATA without issuing source or execution authority."""
    error = "deploy_scene_retirement_runtime_unproven"
    def require(value):
        if not value:
            raise ControlPlaneDeployError(error)
    require(root.is_absolute() and '..' not in root.parts and 0 < len(raw) <= 16*1024*1024
            and 0 < len(proof) <= 32*1024*1024 and time.monotonic() <= deadline)
    def identity(info):
        # Reading immutable proof bytes may update atime. Bind all mutation and
        # ownership metadata, including nanosecond mtime/ctime, instead.
        return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
                info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    def directory(path):
        if path != Path('/'):
            directory(path.parent)
        if not path.exists() and not path.is_symlink():
            path.mkdir(mode=0o755)
        info = path.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid in {0,_SCENE_RUNTIME_OWNER} and not info.st_mode & 0o022)
    directory(root)
    directories = []
    root_fd = os.open('/', os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
    directories.append((root_fd,os.fstat(root_fd)))
    lock = None
    try:
        for part in root.parts[1:]:
            parent = root_fd
            before = os.stat(part,dir_fd=parent,follow_symlinks=False)
            require(stat.S_ISDIR(before.st_mode) and before.st_uid in {0,_SCENE_RUNTIME_OWNER} and not before.st_mode & 0o022)
            root_fd = os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=parent)
            directories.append((root_fd,before))
            require(os.fstat(root_fd) == before)
        lock = os.open('.public-source-proof.lock', os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_CLOEXEC, 0o600,dir_fd=root_fd)
        try:
            info = os.fstat(lock)
            require(stat.S_ISREG(info.st_mode) and info.st_uid == _SCENE_RUNTIME_OWNER and info.st_nlink == 1
                    and stat.S_IMODE(info.st_mode) == 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            # Bind the selected bundle before any fixed manifest/proof bytes.
            # Its protected prefix also pins a retry interrupted in this claim.
            selection = (hashlib.sha256(proof).hexdigest()+'\n').encode()
            for name, raw in zip(('.public-selected-proof.sha256','source-sha256-manifest.json','source-provenance.sigstore.json'), (selection,raw,proof)):
                require(time.monotonic() <= deadline)
                target = root/name
                if target.exists() or target.is_symlink():
                    fd = os.open(target.name, os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=root_fd)
                    try:
                        before = os.fstat(fd)
                        require(stat.S_ISREG(before.st_mode) and before.st_uid == _SCENE_RUNTIME_OWNER
                                and before.st_nlink == 1 and not before.st_mode & 0o022
                                and before.st_size == len(raw) and os.read(fd,len(raw)+1) == raw
                                and identity(before) == identity(os.fstat(fd))
                                == identity(os.stat(target.name,dir_fd=root_fd,follow_symlinks=False)))
                    finally:
                        os.close(fd)
                    continue
                pending = root/(name+'.public-pending')
                flags = os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC
                try:
                    os.stat(pending.name,dir_fd=root_fd,follow_symlinks=False)
                except FileNotFoundError:
                    flags |= os.O_CREAT|os.O_EXCL
                fd = os.open(pending.name, flags, 0o600,dir_fd=root_fd)
                try:
                    before = os.fstat(fd)
                    require(stat.S_ISREG(before.st_mode) and before.st_uid == _SCENE_RUNTIME_OWNER and before.st_nlink == 1
                            and stat.S_IMODE(before.st_mode) in {0o600,0o444} and before.st_size <= len(raw)
                            and (stat.S_IMODE(before.st_mode) != 0o444 or before.st_size == len(raw))
                            and os.read(fd,before.st_size) == raw[:before.st_size])
                    view = memoryview(raw)[before.st_size:]
                    while view:
                        require(time.monotonic() <= deadline)
                        written = os.write(fd, view)
                        require(written > 0)
                        view = view[written:]
                    os.fsync(fd)
                    os.fchmod(fd,0o444)
                    require(os.stat(pending.name,dir_fd=root_fd,follow_symlinks=False).st_ino == before.st_ino)
                    os.link(pending.name,target.name,src_dir_fd=root_fd,dst_dir_fd=root_fd,follow_symlinks=False)
                    os.unlink(pending.name,dir_fd=root_fd)
                finally:
                    os.close(fd)
            os.fsync(root_fd)
            for fd, before in directories:
                current = os.fstat(fd)
                require(stat.S_ISDIR(current.st_mode) and not current.st_mode & 0o022
                        and (current.st_dev,current.st_ino,current.st_mode,current.st_uid,current.st_gid)
                        == (before.st_dev,before.st_ino,before.st_mode,before.st_uid,before.st_gid))
            require(os.fstat(root_fd).st_ino == root.lstat().st_ino)
        finally:
            os.close(lock)
    finally:
        for fd, _ in reversed(directories):
            os.close(fd)

def _scene_source_attestation(source_commit: str, *, deadline: float, _proof_root: Path | None = None) -> tuple[dict[str, Any], bytes, bytes]:
    """Admit signed data with the trusted host tool before executing candidate code."""
    import selectors
    error = "deploy_scene_retirement_runtime_unproven"
    def require(value):
        if not value:
            raise ControlPlaneDeployError(error)
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None and time.monotonic() <= deadline)
    held, identities = [], []
    verifier_cache = None
    def identity(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
                info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    def protected_read(path, cap, *, executable=False):
        require(path.is_absolute() and '..' not in path.parts)
        directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        held.append(directory)
        for part in path.parts[1:-1]:
            info = os.fstat(directory)
            require(stat.S_ISDIR(info.st_mode) and info.st_uid in {0, _SCENE_RUNTIME_OWNER} and not info.st_mode & 0o022)
            identities.append((directory, info))
            directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
            held.append(directory)
        info = os.fstat(directory)
        require(stat.S_ISDIR(info.st_mode) and info.st_uid in {0, _SCENE_RUNTIME_OWNER} and not info.st_mode & 0o022)
        identities.append((directory, info))
        before = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        require(stat.S_ISREG(before.st_mode) and before.st_uid == _SCENE_RUNTIME_OWNER
                and not before.st_mode & 0o022 and before.st_nlink >= 1
                and (executable or before.st_nlink == 1) and 0 < before.st_size <= cap
                and (not executable or before.st_mode & 0o111))
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        held.append(fd)
        require(identity(os.fstat(fd)) == identity(before))
        identities.append((fd, before))
        raw = bytearray()
        if not executable:
            while len(raw) < before.st_size:
                require(time.monotonic() <= deadline)
                block = os.read(fd, min(1024*1024, before.st_size-len(raw)))
                require(block)
                raw.extend(block)
            require(len(raw) == before.st_size and identity(os.fstat(fd)) == identity(before)
                    == identity(os.stat(path.name, dir_fd=directory, follow_symlinks=False)))
        return bytes(raw), fd
    try:
        root = _SCENE_SOURCE_ATTESTATIONS / source_commit if _proof_root is None else _proof_root
        _, gh_fd = protected_read(_SCENE_SOURCE_GH, 256*1024*1024, executable=True)
        if _proof_root is None and not all((root/name).exists() for name in ('source-sha256-manifest.json','source-provenance.sigstore.json')):
            _scene_source_delivery(source_commit, deadline=deadline)
        raw, manifest_fd = protected_read(root / 'source-sha256-manifest.json', 16*1024*1024)
        bundle, bundle_fd = protected_read(root / 'source-provenance.sigstore.json', 32*1024*1024)
        for fd in (manifest_fd, bundle_fd):
            os.lseek(fd, 0, os.SEEK_SET)
        command = [f'/proc/self/fd/{gh_fd}', 'attestation', 'verify', f'/proc/self/fd/{manifest_fd}',
                   '--bundle', f'/proc/self/fd/{bundle_fd}', '--repo', 'ognjhunt/BlueprintCapturePipeline', '--hostname', 'github.com',
                   '--cert-identity', 'https://github.com/ognjhunt/BlueprintCapturePipeline/.github/workflows/ci.yml@refs/heads/main',
                   '--cert-oidc-issuer', 'https://token.actions.githubusercontent.com',
                   '--source-ref', 'refs/heads/main', '--source-digest', source_commit,
                   '--signer-digest', source_commit, '--deny-self-hosted-runners',
                   '--digest-alg', 'sha256', '--predicate-type', 'https://github.com/ognjhunt/BlueprintCapturePipeline/attestations/source-sha256-manifest/v1', '--limit', '1', '--format', 'json']
        # Sigstore initializes authenticated trust metadata in a writable cache.
        # Keep it private and disposable; HOME/config remain credential-free.
        verifier_cache = tempfile.TemporaryDirectory(prefix='blueprint-source-verifier-')
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, pass_fds=(manifest_fd, bundle_fd, gh_fd), start_new_session=True,
            env={'PATH':'/usr/bin:/bin', 'HOME':'/nonexistent', 'GH_CONFIG_DIR':'/nonexistent', 'GH_HOST':'github.com', 'LC_ALL':'C',
                 'XDG_CACHE_HOME':verifier_cache.name})
        output, errors = bytearray(), bytearray()
        verify_deadline = min(deadline, time.monotonic()+120)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, (output, 64*1024*1024))
                selector.register(process.stderr, selectors.EVENT_READ, (errors, 65536))
                while selector.get_map():
                    require(time.monotonic() <= verify_deadline)
                    for key, _ in selector.select(min(.1, max(0, verify_deadline-time.monotonic()))):
                        target, cap = key.data
                        block = os.read(key.fd, min(65536, cap+1-len(target)))
                        if block:
                            require(len(target)+len(block) <= cap)
                            target.extend(block)
                        else:
                            selector.unregister(key.fileobj)
                require(process.wait(timeout=max(.001, verify_deadline-time.monotonic())) == 0)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    process.kill()
            process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()
        for fd, original in identities:
            require(identity(os.fstat(fd)) == identity(original))
        proofs = json.loads(output)
        require(type(proofs) is list and len(proofs) == 1 and type(proofs[0]) is dict)
        verified = proofs[0].get('verificationResult', {})
        require(type(verified) is dict)
        signature = verified.get('signature', {})
        require(type(signature) is dict and type(signature.get('certificate')) is dict and bool(signature['certificate'])
                and type(verified.get('verifiedTimestamps')) is list and bool(verified['verifiedTimestamps'])
                and all(type(value) is dict and value for value in verified['verifiedTimestamps']))
        statement = verified.get('statement', {})
        subjects = statement.get('subject') if type(statement) is dict else None
        manifest_digest = {'sha256':hashlib.sha256(raw).hexdigest()}
        selector_digest = {'sha256':hashlib.sha256(_scene_source_selector_bytes(source_commit)).hexdigest()}
        require(type(statement) is dict and statement.get('_type') == 'https://in-toto.io/Statement/v1'
                and statement.get('predicateType') == 'https://github.com/ognjhunt/BlueprintCapturePipeline/attestations/source-sha256-manifest/v1'
                and type(subjects) is list and len(subjects) in {1,2}
                and all(type(subject) is dict for subject in subjects)
                and sum(subject.get('digest') == manifest_digest for subject in subjects) == 1
                and (len(subjects) == 1 or sum(subject.get('digest') == selector_digest for subject in subjects) == 1)
                and (json.dumps(statement.get('predicate'), sort_keys=True, separators=(',', ':'), ensure_ascii=False)+'\n').encode() == raw)
        manifest = json.loads(raw)
        require(type(manifest) is dict and set(manifest) == {'schema_version', 'sources'}
                and manifest.get('schema_version') == 'blueprint.source_sha256_manifest.v1'
                and type(manifest.get('sources')) is list and len(manifest['sources']) == 2
                and raw == (json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False)+'\n').encode())
        roots = (('src/blueprint_pipeline', 'scripts', 'deploy/systemd', 'uv.lock', 'pyproject.toml'),
                 ('src/blueprint_contracts', 'blueprint_contracts'))
        repositories = ('ognjhunt/BlueprintCapturePipeline', 'ognjhunt/BlueprintContracts')
        commits = (source_commit, '7708a4e4c5dedeeb39cc73d3f6869304de295b81')
        count = total = 0
        for source, repository, commit, wanted in zip(manifest['sources'], repositories, commits, roots):
            require(type(source) is dict and set(source) == {'repository','commit','tree','roots','files'}
                    and source['repository'] == repository and source['commit'] == commit
                    and type(source['tree']) is str and re.fullmatch('[0-9a-f]{40}', source['tree'])
                    and source['roots'] == list(wanted) and type(source['files']) is list and bool(source['files']))
            previous, present = '', set()
            for row in source['files']:
                require(type(row) is dict and set(row) == {'path','git_blob_oid','mode','size','sha256'})
                path = row['path']
                require(type(path) is str and path > previous and len(path.encode()) <= 4096
                        and not path.startswith('/') and '\\' not in path and len(path.split('/')) <= 32
                        and all(part not in {'','.','..'} for part in path.split('/'))
                        and not any(ord(char) < 32 or ord(char) == 127 for char in path)
                        and any(path == root or path.startswith(root+'/') for root in wanted)
                        and row['mode'] in {'100644','100755'} and type(row['git_blob_oid']) is str
                        and re.fullmatch('[0-9a-f]{40}',row['git_blob_oid']) and type(row['sha256']) is str
                        and re.fullmatch('[0-9a-f]{64}',row['sha256']) and type(row['size']) is int
                        and 0 <= row['size'] <= (16*1024*1024 if path == 'uv.lock' else 1024*1024))
                previous = path
                present.add(path)
                count += 1
                total += row['size']
                require(count <= 32768 and total <= 4*1024**3)
            if repository == repositories[0]:
                require(all(any(path == root or path.startswith(root+'/') for path in present) for root in wanted)
                        and {'scripts/install_scene_retirement_runtime.py', 'scripts/deploy_control_plane_commit.py',
                             'scripts/release_source_manifest.py'} <= present)
            else:
                require(any(root+'/__init__.py' in present for root in wanted))
        pipeline = manifest['sources'][0]
        require(pipeline.get('repository') == 'ognjhunt/BlueprintCapturePipeline' and pipeline.get('commit') == source_commit
                and type(pipeline.get('files')) is list and 0 < len(pipeline['files']) <= 32768)
        return manifest, raw, bundle
    finally:
        for fd in reversed(held):
            os.close(fd)
        if verifier_cache is not None:
            verifier_cache.cleanup()

def _bootstrap_scene_retirement_installer(source_repo: Path, source_commit: str, *, deadline: float,
                                         destination: Path | None = None) -> dict[str, Path]:
    """Authenticate installer Git data before the first privileged execution."""
    import fcntl
    import selectors
    error = "deploy_scene_retirement_runtime_unproven"
    shared_deadline = deadline
    def require(value):
        if not value:
            raise ControlPlaneDeployError(error)
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None)
    require(source_repo.is_absolute() and ".." not in source_repo.parts and not source_repo.is_symlink())
    def object_bytes(digest, cap):
        deadline = min(shared_deadline, time.monotonic()+30)
        command = ["/usr/bin/git", "--no-replace-objects", "-C", str(source_repo),
                   "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
                   "-c", "safe.directory=" + str(source_repo), "cat-file", "blob", digest]
        environment = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "GIT_CONFIG_NOSYSTEM": "1",
                       "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_NO_LAZY_FETCH": "1", "GIT_ALLOW_PROTOCOL": "",
                       "GIT_PROTOCOL_FROM_USER": "0", "GIT_TERMINAL_PROMPT": "0"}
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=environment)
        output, errors = bytearray(), bytearray()
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, (output, cap))
                selector.register(process.stderr, selectors.EVENT_READ, (errors, 4096))
                while selector.get_map():
                    require(time.monotonic() <= deadline)
                    for key, _ in selector.select(min(.1, max(0, deadline-time.monotonic()))):
                        buffer, limit = key.data
                        raw = os.read(key.fd, min(65536, limit+1-len(buffer)))
                        if not raw:
                            selector.unregister(key.fileobj)
                        else:
                            require(len(buffer)+len(raw) <= limit)
                            buffer.extend(raw)
            require(process.wait(timeout=max(.001, deadline-time.monotonic())) == 0)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
            process.stderr.close()
        raw = bytes(output)
        return raw
    manifest, manifest_raw, bundle_raw = _scene_source_attestation(source_commit, deadline=shared_deadline)
    files = manifest['sources'][0]['files']
    names = ['scripts/install_scene_retirement_runtime.py', 'scripts/release_source_manifest.py']
    admitted = {}
    for name in names:
        matches = [row for row in files if type(row) is dict and row.get('path') == name]
        require(len(matches) == 1)
        row = matches[0]
        require(set(row) == {'path', 'git_blob_oid', 'mode', 'size', 'sha256'}
                and row['mode'] in {'100644', '100755'} and type(row['size']) is int and 0 < row['size'] <= 1024*1024
                and type(row['git_blob_oid']) is str and re.fullmatch('[0-9a-f]{40}', row['git_blob_oid'])
                and type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']))
        raw = object_bytes(row['git_blob_oid'], row['size'])
        require(len(raw) == row['size'] and hashlib.sha256(raw).hexdigest() == row['sha256'])
        admitted[name] = raw
    body = admitted[names[0]]
    root = destination if destination is not None else _SCENE_RUNTIME_BOOT_ROOT
    def protected_directory(path):
        if not path.exists() and not path.is_symlink():
            protected_directory(path.parent)
            path.mkdir(mode=0o755)
        info = path.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid in {0,_SCENE_RUNTIME_OWNER} and not info.st_mode & 0o022)
        if path != Path('/'):
            protected_directory(path.parent)
    protected_directory(root)
    lock = os.open(root / ".installer-bootstrap.lock",os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
    try:
        info = os.fstat(lock)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == _SCENE_RUNTIME_OWNER and info.st_nlink == 1
                and stat.S_IMODE(info.st_mode) == 0o600)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        target = root / "runtime_installer.py"
        value = json.dumps({"schema":"scene-retirement-runtime-installer.v1",
                 "sha256":"sha256:"+hashlib.sha256(body).hexdigest(),"size_bytes":len(body)},
                 sort_keys=True,separators=(",", ":" )).encode()
        def publish(path, raw):
            if path.exists() or path.is_symlink():
                check = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
                try:
                    old = os.fstat(check)
                    require(stat.S_ISREG(old.st_mode) and old.st_uid == _SCENE_RUNTIME_OWNER
                            and old.st_nlink == 1 and not old.st_mode & 0o022
                            and old.st_size == len(raw) and os.read(check,len(raw)+1) == raw)
                finally:
                    os.close(check)
                return
            partial = path.with_name(path.name+".bootstrap-pending")
            flags = os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC
            if not partial.exists() and not partial.is_symlink():
                flags |= os.O_CREAT|os.O_EXCL
            fd = os.open(partial,flags,0o600)
            try:
                original = os.fstat(fd)
                require(stat.S_ISREG(original.st_mode) and original.st_uid == _SCENE_RUNTIME_OWNER
                        and original.st_nlink == 1 and stat.S_IMODE(original.st_mode) in {0o600,0o644}
                        and original.st_size <= len(raw) and os.read(fd,original.st_size) == raw[:original.st_size])
                view = memoryview(raw)[original.st_size:]
                while view:
                    require(time.monotonic() <= deadline)
                    written = os.write(fd,view)
                    require(written>0)
                    view = view[written:]
                os.fsync(fd)
                os.fchmod(fd,0o644)
                require(partial.lstat().st_ino == original.st_ino)
                os.link(partial,path,follow_symlinks=False)
                os.unlink(partial)
                parent = os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
            finally:
                os.close(fd)
        pending = root / "runtime-installer-pending.json"
        if target.exists() or target.is_symlink():
            require(pending.exists())
            publish(pending,value)
            publish(target,body)
        else:
            publish(pending,value)
            publish(target,body)
        publish(root/"runtime-installer.json",value)
        verifier = admitted[names[1]]
        publish(root/'release_source_manifest.py', verifier)
        publish(root/'source-manifest-verifier.json', json.dumps({
            'schema':'scene-retirement-source-manifest-verifier.v1', 'sha256':hashlib.sha256(verifier).hexdigest(),
            'size':len(verifier)}, sort_keys=True, separators=(',', ':')).encode())
        publish(root/'source-sha256-manifest.json', manifest_raw)
        publish(root/'source-provenance.sigstore.json', bundle_raw)
        return {'source_manifest':root/'source-sha256-manifest.json',
                'source_attestation':root/'source-provenance.sigstore.json', 'manifest_verifier':root/'release_source_manifest.py'}
    finally:
        os.close(lock)

def _scene_runtime_diagnostic(stderr: bytes | str | None, *, phase: str, reason: str) -> dict[str, str]:
    """Admit only bounded fixed markers; child output is never forwarded."""
    phases = {"source_attestation", "signed_release", "build_sdk", "resume_initial_intent", "refresh", "prepare", "publish_installer"}
    reasons = {"deadline", "validation", "io", "unexpected"}
    if isinstance(stderr, (bytes, str)) and len(stderr) <= 65536:
        text = stderr.decode("ascii", errors="replace") if isinstance(stderr, bytes) else stderr
        for line in text.splitlines():
            if line.startswith("scene_retirement_runtime_phase:") and line.split(":", 1)[1] in phases:
                phase = line.split(":", 1)[1]
            if line.startswith("scene_retirement_runtime_failure:") and line.split(":", 1)[1] in reasons:
                reason = line.split(":", 1)[1]
    return {"phase": phase, "reason": reason}

def _prepare_scene_retirement_runtime(*, source_repo: Path, source_commit: str, _deadline: float | None = None) -> dict[str, Any]:
    """Authenticate the selected release installer before exposing new units."""
    # One installation-only origin covers authentication, SDK and durable copy.
    started = time.monotonic()
    if _deadline is not None and (type(_deadline) is not float or not math.isfinite(_deadline)):
        raise ControlPlaneDeployError("deploy_scene_retirement_runtime_unproven")
    deadline = started + _SCENE_RUNTIME_INSTALL_SECONDS if _deadline is None else min(started + _SCENE_RUNTIME_INSTALL_SECONDS, _deadline)
    if started > deadline:
        raise ControlPlaneDeployError("deploy_scene_retirement_runtime_unproven")
    root = _SCENE_RUNTIME_BOOT_ROOT
    helper = root / "runtime_installer.py"
    error = "deploy_scene_retirement_runtime_unproven"
    phase = "retained_installer"
    held: list[int] = []
    def identity(info: os.stat_result) -> tuple[int, ...]:
        return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    def read(path: Path, cap: int) -> tuple[bytes, int]:
        for ancestor in (*reversed(path.parent.parents), path.parent):
            info = ancestor.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid not in {0, _SCENE_RUNTIME_OWNER} or info.st_mode & 0o022:
                raise ControlPlaneDeployError(error)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        held.append(fd)
        first = os.fstat(fd)
        if (not stat.S_ISREG(first.st_mode) or first.st_uid != _SCENE_RUNTIME_OWNER or first.st_mode & 0o022
                or first.st_nlink != 1 or first.st_size > cap):
            raise ControlPlaneDeployError(error)
        raw = os.read(fd, cap + 1)
        if len(raw) != first.st_size or identity(os.fstat(fd)) != identity(first) or identity(path.lstat()) != identity(first):
            raise ControlPlaneDeployError(error)
        return raw, fd
    def verified_installer(directory: Path) -> int:
        raw, fd = read(directory / "runtime_installer.py", 1024 * 1024)
        selected = {"sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
        matched = False
        for name in ("runtime-installer.json", "runtime-installer-pending.json"):
            path = directory / name
            if path.exists() or path.is_symlink():
                body, _ = read(path, 4096)
                value = json.loads(body)
                if (type(value) is dict and set(value) == {"schema", "sha256", "size_bytes"}
                        and value["schema"] == "scene-retirement-runtime-installer.v1"
                        and {key: value[key] for key in selected} == selected):
                    matched = True
        if not matched or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
            raise ControlPlaneDeployError(error)
        return fd
    try:
        # Unknown retained state still refuses before staging or execution. A
        # valid old helper must not prevent a reviewed installer repair from
        # running: stage authenticated Git bytes separately and preserve it.
        retained = (helper, root / "runtime-installer.json", root / "runtime-installer-pending.json")
        if any(path.exists() or path.is_symlink() for path in retained):
            verified_installer(root)
        candidate = root / "installers" / source_commit
        phase = "authenticate_installer"
        attestation = _bootstrap_scene_retirement_installer(
            source_repo, source_commit, deadline=deadline, destination=candidate,
        )
        fd = verified_installer(candidate)
        os.lseek(fd, 0, os.SEEK_SET)
        command = ["/usr/bin/python3", "-I", "-S", f"/proc/self/fd/{fd}",
                   "--source", str(source_repo), "--source-commit", source_commit, "--locked-sdk",
                   "--deadline-monotonic", str(deadline),
                   "--source-manifest", str(attestation['source_manifest']),
                   "--source-attestation", str(attestation['source_attestation']),
                   "--manifest-verifier", str(attestation['manifest_verifier'])]
        phase = "execute_installer"
        result = subprocess.run(command, pass_fds=(fd,), stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=max(.001, deadline-time.monotonic()), check=False,
                                env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LC_ALL": "C"})
        if result.returncode != 0 or len(result.stdout) > 65536 or len(result.stderr) > 65536:
            failure = ControlPlaneDeployError(error)
            setattr(failure, "runtime_diagnostic", _scene_runtime_diagnostic(result.stderr, phase=phase, reason="validation"))
            raise failure
        phase = "verify_result"
        value = json.loads(result.stdout)
        if (type(value) is not dict or value.get("status") not in {"prepared", "refreshed"}
                or value.get("source_commit") != source_commit
                or value.get("authority_issued") is not False or value.get("cleanup_enabled") is not False):
            raise ControlPlaneDeployError(error)
        return value
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        failure = ControlPlaneDeployError(error)
        diagnostic = getattr(exc, "runtime_diagnostic", None)
        if diagnostic is None:
            timed_out = isinstance(exc, subprocess.TimeoutExpired)
            diagnostic = _scene_runtime_diagnostic(
                exc.stderr if isinstance(exc, subprocess.TimeoutExpired) else None, phase=phase,
                reason="deadline" if timed_out else "io" if isinstance(exc, OSError) else "validation",
            )
            if timed_out:
                diagnostic["reason"] = "deadline"
        setattr(failure, "runtime_diagnostic", diagnostic)
        raise failure from exc
    finally:
        for fd in reversed(held):
            os.close(fd)

# One origin precedes Git selection and all proof acquisition; never renew it.
deadline = time.monotonic() + _SCENE_RUNTIME_INSTALL_SECONDS
source = Path(sys.argv[1])
commit = subprocess.check_output(['/usr/bin/git','--no-replace-objects','-C',str(source),'-c','core.hooksPath=/dev/null',
                                 '-c','core.fsmonitor=false','-c','safe.directory='+str(source),'rev-parse','HEAD'],
                                 env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent','GIT_CONFIG_NOSYSTEM':'1',
                                      'GIT_CONFIG_GLOBAL':'/dev/null'},timeout=min(10,max(.001,deadline-time.monotonic()))).decode().strip()
print(json.dumps(_prepare_scene_retirement_runtime(source_repo=source,source_commit=commit,_deadline=deadline),sort_keys=True))

PY_RUNTIME
fi

# The service account runs git against this checkout to pin the allocator's
# source identity. A root-owned checkout makes git refuse with "detected
# dubious ownership", the identity probe fails, and a paid launch is rejected
# at admission -- so the account that reads the repository must own it.
run chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${REPO_ROOT}"

run install -d -m 0755 "${SYSTEMD_DIR}"
run install -d -m 0750 -o root -g "${SERVICE_GROUP}" "${ENV_DIR}"
run install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
  "${HANDOFF_DIR}"
run install -d -m 0750 -o root -g "${SERVICE_GROUP}" \
  "${PROVIDER_SECRETS_DIR}"
run install -d -m 0750 -o root -g "${SERVICE_GROUP}" \
  "${CREDENTIALS_DIR}"
run install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
  "${LAUNCH_PROFILE_DIR}"
run install -d -m 0750 -o root -g "${SERVICE_GROUP}" \
  "${CAPTURE_RECONSTRUCTION_POLICY_DIR}"
run install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
  "${STATE_DIR}" \
  "${STATE_DIR}/robot-eval-job-requests" \
  "${STATE_DIR}/incoming_webapp_job_requests" \
  "${STATE_DIR}/deliveries" \
  "${STATE_DIR}/gpu_spend_guard" \
  "${STATE_DIR}/provider-locks" \
  "${STATE_DIR}/storage-pins"
run install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
  "${STATE_DIR}/task-evaluation-launches/pending" \
  "${STATE_DIR}/task-evaluation-launches/processing" \
  "${STATE_DIR}/task-evaluation-launches/completed" \
  "${STATE_DIR}/task-evaluation-launches/blocked" \
  "${STATE_DIR}/task-evaluation-launch-runs" \
  "${STATE_DIR}/task-evaluation-terminal-resource-releases/pending" \
  "${STATE_DIR}/task-evaluation-terminal-resource-releases/processing" \
  "${STATE_DIR}/task-evaluation-terminal-resource-releases/completed" \
  "${STATE_DIR}/task-evaluation-terminal-resource-releases/blocked" \
  "${STATE_DIR}/task-evaluation-terminal-resource-releases-state" \
  "${STATE_DIR}/task-evaluation-control-plane-releases" \
  "${STATE_DIR}/task-evaluation-launch-reconciliation" \
  "${STATE_DIR}/task-evaluation-launch-supervision/recommendations"
run install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
  "${STATE_DIR}/task-evaluation-launch-preparations" \
  "${STATE_DIR}/task-evaluation-launch-preparations/pending" \
  "${STATE_DIR}/task-evaluation-launch-preparations/processing" \
  "${STATE_DIR}/task-evaluation-launch-preparations/materialized" \
  "${STATE_DIR}/task-evaluation-launch-preparations/blocked" \
  "${STATE_DIR}/task-evaluation-launch-preparations/identities" \
  "${STATE_DIR}/task-evaluation-launch-preparations/results" \
  "${STATE_DIR}/scene-object-discoveries/pending" \
  "${STATE_DIR}/scene-object-discoveries/processing" \
  "${STATE_DIR}/scene-object-discoveries/blocked" \
  "${STATE_DIR}/scene-object-discoveries/results" \
  "${STATE_DIR}/scene-object-discoveries/identities" \
  "${STATE_DIR}/scene-object-discoveries/selections" \
  "${STATE_DIR}/task-evaluation-scene-constructions/pending" \
  "${STATE_DIR}/task-evaluation-scene-constructions/processing" \
  "${STATE_DIR}/task-evaluation-scene-constructions/completed" \
  "${STATE_DIR}/task-evaluation-scene-constructions/blocked" \
  "${STATE_DIR}/task-evaluation-scene-constructions/results" \
  "${STATE_DIR}/task-evaluation-episode-compilations/pending" \
  "${STATE_DIR}/task-evaluation-episode-compilations/processing" \
  "${STATE_DIR}/task-evaluation-episode-compilations/completed" \
  "${STATE_DIR}/task-evaluation-episode-compilations/blocked" \
  "${STATE_DIR}/task-evaluation-episode-compilations/results" \
  "${STATE_DIR}/remote-cpu-jobs" \
  "${STATE_DIR}/remote-cpu-jobs/handoffs/episode_compilation" \
  "${STATE_DIR}/remote-cpu-jobs/shadow/episode_compilation" \
  "${STATE_DIR}/remote-cpu-jobs/fallback/episode_compilation" \
  "${STATE_DIR}/remote-cpu-jobs/recovery/episode_compilation" \
  "${STATE_DIR}/task-evaluation-launch-activations" \
  "${STATE_DIR}/task-evaluation-launch-activations/pending" \
  "${STATE_DIR}/task-evaluation-launch-activations/processing" \
  "${STATE_DIR}/task-evaluation-launch-activations/prepared" \
  "${STATE_DIR}/task-evaluation-launch-activations/blocked" \
  "${STATE_DIR}/task-evaluation-launch-activations/identities" \
  "${STATE_DIR}/task-evaluation-launch-activations/results" \
  "${STATE_DIR}/task-evaluation-policy-canary-dispatches" \
  "${STATE_DIR}/task-evaluation-policy-canary-dispatches/pending" \
  "${STATE_DIR}/task-evaluation-policy-canary-dispatches/processing" \
  "${STATE_DIR}/task-evaluation-policy-canary-dispatches/completed" \
  "${STATE_DIR}/task-evaluation-policy-canary-dispatches/blocked" \
  "${STATE_DIR}/task-evaluation-policy-canaries" \
  "${STATE_DIR}/standing-authorizations" \
  "${TASK_EVALUATION_INPUT_ROOT}" \
  "${TASK_EVALUATION_INPUT_ROOT}/prepared-references" \
  "${TASK_EVALUATION_INPUT_ROOT}/compiled-episodes" \
  "${TASK_EVALUATION_INPUT_ROOT}/launch-activations" \
  "${TASK_EVALUATION_INPUT_ROOT}/lanes" \
  "${TASK_EVALUATION_INPUT_ROOT}/policy-canary-execution-setups" \
  "${TASK_EVALUATION_INPUT_ROOT}/system-runtimes"
run install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
  "${WORK_VOLUME_ROOT}/lanes"
run install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
  "${TASK_EVALUATION_INPUT_ROOT}/scene-object-discoveries" \
  "${TASK_EVALUATION_INPUT_ROOT}/scene-object-discovery-outputs"
if [[ "${DRY_RUN}" == "true" ]]; then
  printf '[dry-run] verify mode=0750 owner=%s group=%s %s\n' \
    "${SERVICE_USER}" "${SERVICE_GROUP}" "${CONFIGURED_CONTROLS_PLAN_ROOT}"
  printf '[dry-run] verify mode=0750 owner=root group=%s %s\n' \
    "${SERVICE_GROUP}" "${CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT}"
  printf '[dry-run] verify mode=0440 owner=root group=%s %s\n' \
    "${SERVICE_GROUP}" "${CONFIGURED_CONTROLS_WEBAPP_SECRET}"
else
  if [[ -L "${CONFIGURED_CONTROLS_PLAN_ROOT}" ]]; then
    echo "ERROR: configured-controls plan root is a symlink" >&2
    exit 1
  fi
  if [[ ! -d "${CONFIGURED_CONTROLS_PLAN_ROOT}" ]]; then
    run mkdir -p "${CONFIGURED_CONTROLS_PLAN_ROOT}"
  fi
  PLAN_OWNER="$(stat -c '%U:%G' "${CONFIGURED_CONTROLS_PLAN_ROOT}")"
  PLAN_MODE="$(stat -c '%a' "${CONFIGURED_CONTROLS_PLAN_ROOT}")"
  if [[ "${PLAN_OWNER}" != "${SERVICE_USER}:${SERVICE_GROUP}" ]]; then
    run chown "${SERVICE_USER}:${SERVICE_GROUP}" "${CONFIGURED_CONTROLS_PLAN_ROOT}"
  fi
  if [[ "${PLAN_MODE}" != "750" ]]; then
    run chmod 0750 "${CONFIGURED_CONTROLS_PLAN_ROOT}"
  fi
  test "$(stat -c '%U:%G:%a' "${CONFIGURED_CONTROLS_PLAN_ROOT}")" = \
    "${SERVICE_USER}:${SERVICE_GROUP}:750"
  if [[ -L "${CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT}" ]]; then
    echo "ERROR: configured-controls autostart intent root is a symlink" >&2
    exit 1
  fi
  if [[ ! -d "${CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT}" ]]; then
    run mkdir -p "${CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT}"
  fi
  run chown "root:${SERVICE_GROUP}" "${CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT}"
  run chmod 0750 "${CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT}"
  test "$(stat -c '%U:%G:%a' "${CONFIGURED_CONTROLS_AUTOSTART_INTENT_ROOT}")" = \
    "root:${SERVICE_GROUP}:750"
  if [[ ! -f "${CONFIGURED_CONTROLS_WEBAPP_SECRET}" || -L "${CONFIGURED_CONTROLS_WEBAPP_SECRET}" ]]; then
    echo "ERROR: configured-controls WebApp submit secret missing or unsafe" >&2
    exit 1
  fi
  SECRET_OWNER="$(stat -c '%U:%G' "${CONFIGURED_CONTROLS_WEBAPP_SECRET}")"
  SECRET_MODE="$(stat -c '%a' "${CONFIGURED_CONTROLS_WEBAPP_SECRET}")"
  if [[ "${SECRET_OWNER}" != "root:${SERVICE_GROUP}" ]]; then
    run chown "root:${SERVICE_GROUP}" "${CONFIGURED_CONTROLS_WEBAPP_SECRET}"
  fi
  if [[ "${SECRET_MODE}" != "440" ]]; then
    run chmod 0440 "${CONFIGURED_CONTROLS_WEBAPP_SECRET}"
  fi
  test "$(stat -c '%U:%G:%a' "${CONFIGURED_CONTROLS_WEBAPP_SECRET}")" = \
    "root:${SERVICE_GROUP}:440"
fi
run install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
  "${STATE_DIR}/capture-reconstruction-queue/pending" \
  "${STATE_DIR}/capture-reconstruction-queue/processing" \
  "${STATE_DIR}/capture-reconstruction-queue/completed" \
  "${STATE_DIR}/capture-reconstruction-queue/blocked" \
  "${STATE_DIR}/capture-reconstruction-runs" \
  "${STATE_DIR}/capture-reconstruction-derived"
# Older units ran as root. Migrate only the two explicitly bounded runtime
# trees before installing the hardened service-user units. GNU chown's
# --no-dereference keeps a symlink itself in scope instead of following it to
# an unrelated target.
run chown -R --no-dereference "${SERVICE_USER}:${SERVICE_GROUP}" \
  "${HANDOFF_DIR}" \
  "${STATE_DIR}"
# BEGIN scene retirement stores
# Outside STATE_DIR: its legacy recursive ownership migration must never grant
# service ownership of private consent, archive or recovery records. Provision
# storage only; installing these directories does not issue or enable a policy.
SCENE_RETIREMENT_ROOT="/var/lib/blueprint/scene-retirement"
SCENE_RETIREMENT_STORES=(
  "${SCENE_RETIREMENT_ROOT}"
  "${SCENE_RETIREMENT_ROOT}/coordinator"
  "${SCENE_RETIREMENT_ROOT}/generations"
  "${SCENE_RETIREMENT_ROOT}/journals"
  "${SCENE_RETIREMENT_ROOT}/journals/processes"
  "${SCENE_RETIREMENT_ROOT}/journals/retired"
  "${SCENE_RETIREMENT_ROOT}/journals.metadata"
  "${SCENE_RETIREMENT_ROOT}/consents"
)
SCENE_RETIREMENT_MODES=(755 755 700 700 700 700 750 700)
SCENE_RETIREMENT_OWNERS=(root root "${SERVICE_USER}" root root root root root)
# Preflight every named directory and ancestry before the first mutation.
# Existing authority is validated, never repaired or recursively re-owned.
for SCENE_INDEX in "${!SCENE_RETIREMENT_STORES[@]}"; do
  SCENE_DIRECTORY="${SCENE_RETIREMENT_STORES[SCENE_INDEX]}"
  SCENE_ANCESTOR="${SCENE_DIRECTORY}"
  while [[ "${SCENE_ANCESTOR}" != / ]]; do
    if [[ -L "${SCENE_ANCESTOR}" ]] || \
       [[ -e "${SCENE_ANCESTOR}" && ! -d "${SCENE_ANCESTOR}" ]]; then
      echo "ERROR: scene retirement storage contains a linked or non-directory component" >&2
      exit 1
    fi
    if [[ "${DRY_RUN}" != true && -d "${SCENE_ANCESTOR}" && \
          "${SCENE_ANCESTOR}" != "${SCENE_DIRECTORY}" ]]; then
      SCENE_PARENT_UID="$(stat -c '%u' "${SCENE_ANCESTOR}")"
      SCENE_PARENT_MODE="$(stat -c '%a' "${SCENE_ANCESTOR}")"
      if [[ "${SCENE_PARENT_UID}" != 0 ]] || (( (8#${SCENE_PARENT_MODE} & 8#022) != 0 )); then
        echo "ERROR: scene retirement storage ancestry is not root-protected" >&2
        exit 1
      fi
    fi
    SCENE_ANCESTOR="$(dirname -- "${SCENE_ANCESTOR}")"
  done
  if [[ "${DRY_RUN}" != true && -d "${SCENE_DIRECTORY}" ]]; then
    SCENE_EXPECTED="${SCENE_RETIREMENT_OWNERS[SCENE_INDEX]}:${SERVICE_GROUP}:${SCENE_RETIREMENT_MODES[SCENE_INDEX]}"
    if [[ "$(stat -c '%U:%G:%a' "${SCENE_DIRECTORY}")" != "${SCENE_EXPECTED}" ]]; then
      echo "ERROR: existing scene retirement store has unexpected ownership or permissions" >&2
      exit 1
    fi
  fi
done
run install -d -m 0755 -o root -g "${SERVICE_GROUP}" "${SCENE_RETIREMENT_ROOT}"
run install -d -m 0755 -o root -g "${SERVICE_GROUP}" "${SCENE_RETIREMENT_ROOT}/coordinator"
run install -d -m 0700 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" "${SCENE_RETIREMENT_ROOT}/generations"
run install -d -m 0700 -o root -g "${SERVICE_GROUP}" "${SCENE_RETIREMENT_ROOT}/journals"
run install -d -m 0700 -o root -g "${SERVICE_GROUP}" "${SCENE_RETIREMENT_ROOT}/journals/processes"
run install -d -m 0700 -o root -g "${SERVICE_GROUP}" "${SCENE_RETIREMENT_ROOT}/journals/retired"
run install -d -m 0750 -o root -g "${SERVICE_GROUP}" "${SCENE_RETIREMENT_ROOT}/journals.metadata"
run install -d -m 0700 -o root -g "${SERVICE_GROUP}" "${SCENE_RETIREMENT_ROOT}/consents"
# END scene retirement stores
# Host hygiene that used to be hand-applied: bound journald and age /var/tmp.
run install -d -m 0755 /etc/systemd/journald.conf.d /etc/tmpfiles.d
run install -m 0644 \
  "${REPO_ROOT}/deploy/host/journald.conf.d/50-blueprint-cap.conf" \
  /etc/systemd/journald.conf.d/50-blueprint-cap.conf
run install -m 0644 \
  "${REPO_ROOT}/deploy/host/tmpfiles.d/blueprint-var-tmp.conf" \
  /etc/tmpfiles.d/blueprint-var-tmp.conf
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-pipeline-control-plane.service" \
  "${SYSTEMD_DIR}/blueprint-pipeline-control-plane.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-agent-run-dispatcher.service" \
  "${SYSTEMD_DIR}/blueprint-agent-run-dispatcher.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-agent-run-dispatcher.timer" \
  "${SYSTEMD_DIR}/blueprint-agent-run-dispatcher.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-pipeline-control-plane.timer" \
  "${SYSTEMD_DIR}/blueprint-pipeline-control-plane.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-pipeline-intake.service" \
  "${SYSTEMD_DIR}/blueprint-pipeline-intake.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-pubsub-handoff-listener.service" \
  "${SYSTEMD_DIR}/blueprint-pubsub-handoff-listener.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-pubsub-handoff-listener.timer" \
  "${SYSTEMD_DIR}/blueprint-pubsub-handoff-listener.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-capture-reconstruction-dispatcher.service" \
  "${SYSTEMD_DIR}/blueprint-capture-reconstruction-dispatcher.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-capture-reconstruction-dispatcher.path" \
  "${SYSTEMD_DIR}/blueprint-capture-reconstruction-dispatcher.path"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-capture-reconstruction-dispatcher.timer" \
  "${SYSTEMD_DIR}/blueprint-capture-reconstruction-dispatcher.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-gpu-spend-guard.service" \
  "${SYSTEMD_DIR}/blueprint-gpu-spend-guard.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-gpu-spend-guard.timer" \
  "${SYSTEMD_DIR}/blueprint-gpu-spend-guard.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-launch-dispatcher.service" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-launch-dispatcher.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-launch-dispatcher.path" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-launch-dispatcher.path"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-launch-preparation.service" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-launch-preparation.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-launch-preparation.path" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-launch-preparation.path"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-launch-preparation.timer" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-launch-preparation.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-scene-object-discovery.service" \
  "${SYSTEMD_DIR}/blueprint-scene-object-discovery.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-scene-object-discovery.path" \
  "${SYSTEMD_DIR}/blueprint-scene-object-discovery.path"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-episode-compilation.service" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-episode-compilation.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-episode-compilation.path" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-episode-compilation.path"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-episode-compilation.timer" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-episode-compilation.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-episode-compilation-remote.service" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-episode-compilation-remote.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-episode-compilation-remote.timer" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-episode-compilation-remote.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-episode-compilation-remote.path" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-episode-compilation-remote.path"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-launch-activation.service" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-launch-activation.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-launch-activation.path" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-launch-activation.path"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-policy-canary-dispatcher.service" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-policy-canary-dispatcher.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-policy-canary-dispatcher.path" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-policy-canary-dispatcher.path"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-native-g1-team-campaign-dispatcher.service" \
  "${SYSTEMD_DIR}/blueprint-native-g1-team-campaign-dispatcher.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-native-g1-team-campaign-dispatcher.timer" \
  "${SYSTEMD_DIR}/blueprint-native-g1-team-campaign-dispatcher.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-native-g1-team-campaign-settlement.service" \
  "${SYSTEMD_DIR}/blueprint-native-g1-team-campaign-settlement.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-native-g1-team-campaign-settlement.timer" \
  "${SYSTEMD_DIR}/blueprint-native-g1-team-campaign-settlement.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-terminal-resource-release.service" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-terminal-resource-release.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-terminal-resource-release.path" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-terminal-resource-release.path"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-launch-reconciler.service" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-launch-reconciler.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-launch-reconciler.timer" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-launch-reconciler.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-configured-controls-progression.service" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-configured-controls-progression.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-scene-progression.service" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-scene-progression.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-scene-progression.timer" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-scene-progression.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-configured-controls-progression.timer" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-configured-controls-progression.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-configured-controls-progression.path" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-configured-controls-progression.path"
install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-control-plane-storage-gc.service" \
  "${SYSTEMD_DIR}/blueprint-control-plane-storage-gc.service"
install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-control-plane-storage-gc.timer" \
  "${SYSTEMD_DIR}/blueprint-control-plane-storage-gc.timer"
install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-control-plane-capacity.service" \
  "${SYSTEMD_DIR}/blueprint-control-plane-capacity.service"
install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-control-plane-capacity.timer" \
  "${SYSTEMD_DIR}/blueprint-control-plane-capacity.timer"
install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-control-plane-preflight.service" \
  "${SYSTEMD_DIR}/blueprint-control-plane-preflight.service"
install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-control-plane-preflight.timer" \
  "${SYSTEMD_DIR}/blueprint-control-plane-preflight.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-launch-supervisor.service" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-launch-supervisor.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-task-evaluation-launch-supervisor.timer" \
  "${SYSTEMD_DIR}/blueprint-task-evaluation-launch-supervisor.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-provider-billing-reconciler.service" \
  "${SYSTEMD_DIR}/blueprint-provider-billing-reconciler.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-provider-billing-reconciler.timer" \
  "${SYSTEMD_DIR}/blueprint-provider-billing-reconciler.timer"

if [[ ! -f "${ENV_FILE}" ]]; then
  run install -o root -g "${SERVICE_GROUP}" -m 0640 \
    "${REPO_ROOT}/deploy/systemd/pipeline-control-plane.env.example" \
    "${ENV_FILE}"
  echo "created ${ENV_FILE}; fill secrets before enabling live actions"
else
  run chown root:"${SERVICE_GROUP}" "${ENV_FILE}"
  run chmod 0640 "${ENV_FILE}"
  echo "kept existing ${ENV_FILE}"
fi

# Record the deployed source identity.  The intake service reports
# commit_proven=false and answers /api/live-pipeline/version with 503 until
# BLUEPRINT_SOURCE_COMMIT names the exact deployed commit, so an otherwise
# healthy host looks unprovable without it.  This is an identity fact, not a
# secret, and it is refreshed on every install so it cannot drift behind the
# checkout the services actually import.
if SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null)"; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run] record BLUEPRINT_SOURCE_COMMIT=%s\n' "${SOURCE_COMMIT}"
  else
    sed -i '/^BLUEPRINT_SOURCE_COMMIT=/d' "${ENV_FILE}"
    printf 'BLUEPRINT_SOURCE_COMMIT=%s\n' "${SOURCE_COMMIT}" >> "${ENV_FILE}"
    echo "recorded deployed source commit ${SOURCE_COMMIT}"
  fi
else
  echo "WARNING: no source commit resolved; /api/live-pipeline/version stays 503" >&2
fi

# The production host intentionally retains one service virtualenv across
# promoted checkouts. Merely moving that checkout does not install a newly
# declared dependency, which would let the units restart on source they cannot
# import. Keep the small control-plane-only delta hash-pinned and install it as
# the unprivileged service owner before any worker wheel or unit is activated.
RUNTIME_REQUIREMENTS="${REPO_ROOT}/deploy/systemd/production-control-plane-requirements.txt"
RUNTIME_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ "${DRY_RUN}" == "true" ]]; then
  printf '[dry-run] synchronize hash-pinned production runtime requirements from %s\n' \
    "${RUNTIME_REQUIREMENTS}"
else
  if [[ ! -x "${RUNTIME_PYTHON}" ]]; then
    echo "ERROR: production service virtualenv is missing at ${RUNTIME_PYTHON}" >&2
    exit 1
  fi
  # pip leaves exact installed pins alone. Checking only rfc8785 would skip
  # newly added compiler pins on an otherwise provisioned host.
  run runuser -u "${SERVICE_USER}" -- "${RUNTIME_PYTHON}" -m pip install \
    --disable-pip-version-check --no-deps --only-binary=:all: \
    --require-hashes --requirement "${RUNTIME_REQUIREMENTS}"
  run runuser -u "${SERVICE_USER}" -- "${RUNTIME_PYTHON}" -c \
    'import importlib.metadata as m; assert m.version("rfc8785") == "0.1.4"'
  echo "synchronized hash-pinned production runtime dependencies"
fi

# Build the Windows worker package from the exact promoted commit and bind the
# environment to that immutable release directory. The transport compiler also
# verifies the wheel's embedded source digest before any provider mutation.
# This closes the gap where an old wheel at a stable placeholder path could be
# paired with a newer dispatcher checkout.
if [[ -n "${SOURCE_COMMIT:-}" ]]; then
  WORKER_RELEASE_DIR="/opt/blueprint/releases/canonical-3dgs-worker/${SOURCE_COMMIT}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run] build exact worker wheel for %s into %s\n' \
      "${SOURCE_COMMIT}" "${WORKER_RELEASE_DIR}"
  else
    run install -d -m 0755 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
      "${WORKER_RELEASE_DIR}"
    WORKER_BUILD_RECEIPT="$(runuser -u "${SERVICE_USER}" -- \
      "${REPO_ROOT}/scripts/build_canonical_3dgs_worker_wheel.sh" \
      "${SOURCE_COMMIT}" "${WORKER_RELEASE_DIR}")"
    WORKER_WHEEL="$(printf '%s' "${WORKER_BUILD_RECEIPT}" | \
      python3 -c 'import json,sys; print(json.load(sys.stdin)["wheel_path"])')"
    if [[ ! -f "${WORKER_WHEEL}" ]]; then
      echo "ERROR: exact canonical worker wheel was not produced" >&2
      exit 1
    fi
    printf '%s\n' "${WORKER_BUILD_RECEIPT}" > \
      "${WORKER_RELEASE_DIR}/build_receipt.json"
    run chown root:"${SERVICE_GROUP}" \
      "${WORKER_RELEASE_DIR}/build_receipt.json"
    run chmod 0644 "${WORKER_RELEASE_DIR}/build_receipt.json"
    sed -i '/^BLUEPRINT_CANONICAL_3DGS_WORKER_WHEEL=/d' "${ENV_FILE}"
    printf 'BLUEPRINT_CANONICAL_3DGS_WORKER_WHEEL=%s\n' "${WORKER_WHEEL}" >> \
      "${ENV_FILE}"
    echo "bound exact canonical worker wheel ${WORKER_WHEEL}"
  fi
fi

# Single-use paid-attempt enforcement writes a consumption record before any
# provider allocation.  It defaults to the invoking user's home, which the
# hardened units cannot reach: the service account's home is /nonexistent and
# the units set ProtectHome=true.  Without this the ledger is unwritable and
# every paid run fails *after* its authority validates, which reads as a
# spend-authority fault rather than a filesystem one.  Bind it to the state
# directory the units already grant in ReadWritePaths.
SPEND_AUTHORITY_ROOT="${SPEND_AUTHORITY_ROOT:-/var/lib/blueprint/spend-authority}"
if [[ "${DRY_RUN}" == "true" ]]; then
  printf '[dry-run] record BLUEPRINT_SPEND_AUTHORITY_ROOT=%s\n' "${SPEND_AUTHORITY_ROOT}"
else
  run mkdir -p "${SPEND_AUTHORITY_ROOT}"
  run chown "${SERVICE_USER}":"${SERVICE_GROUP}" "${SPEND_AUTHORITY_ROOT}"
  # The consumption check refuses a group- or world-accessible tree, because a
  # second writer could forge or delete a record and re-fund an allocation.
  run chmod 0700 "${SPEND_AUTHORITY_ROOT}"
  # Plan 14 §1: the remote episode-compilation unit writes only these two ledgers, so they exist up front.
  run install -d -m 0700 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
    "${SPEND_AUTHORITY_ROOT}/consumed" "${SPEND_AUTHORITY_ROOT}/remote-cpu-settled"
  sed -i '/^BLUEPRINT_SPEND_AUTHORITY_ROOT=/d' "${ENV_FILE}"
  printf 'BLUEPRINT_SPEND_AUTHORITY_ROOT=%s\n' "${SPEND_AUTHORITY_ROOT}" >> "${ENV_FILE}"
  echo "bound spend-authority ledger to ${SPEND_AUTHORITY_ROOT}"

  # Same interpreter selection the units use, so the installer and the running
  # service agree on which checkout's code reconciles the ledger.
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    LEDGER_PYTHON="${REPO_ROOT}/.venv/bin/python"
  else
    LEDGER_PYTHON="$(command -v python3)"
  fi

  # Binding the root on a host that was already running moves the ledger and
  # leaves its consumption records at the previous location, so the new root
  # reads empty and every authorization spent there looks unspent.  Adopt them
  # now rather than letting the unit discover it at its next paid attempt.  The
  # startup guard repeats this check, so a host rebuilt without the installer is
  # still covered; running it here surfaces the failure while an operator is
  # watching.
  # Run as the service account so adopted records carry the ownership the
  # consumption check requires; root-owned records would be refused.
  if runuser -u "${SERVICE_USER}" -- env \
       BLUEPRINT_SPEND_AUTHORITY_ROOT="${SPEND_AUTHORITY_ROOT}" \
       PYTHONPATH="${REPO_ROOT}/src" \
       "${LEDGER_PYTHON}" -m blueprint_pipeline.spend_authority_ledger_migration \
       --receipt-out "${SPEND_AUTHORITY_ROOT}/reconciliation_receipt.json"; then
    echo "reconciled spend-authority ledger"
  else
    echo "ERROR: spend-authority ledger could not be reconciled; refusing to continue" >&2
    echo "       a ledger stranded at a previous root disables single-use spend enforcement" >&2
    exit 1
  fi
fi

# Render the public TLS/reverse-proxy edge.  The intake service binds loopback
# only, so without this the control plane has no reachable surface at all.
# The hostname stays operator-supplied because a rebuilt host gets a new
# address; refusing to guess is what keeps a dead name out of the config.
if [[ -n "${BLUEPRINT_PIPELINE_PUBLIC_HOSTNAME:-}" ]]; then
  run install -d -m 0755 "$(dirname "${CADDY_SITE_FILE}")"
  run install -m 0644 "${REPO_ROOT}/deploy/caddy/Caddyfile" "${CADDY_SITE_FILE}"
  echo "installed caddy edge config at ${CADDY_SITE_FILE}"
  echo "  serving ${BLUEPRINT_PIPELINE_PUBLIC_HOSTNAME}; reload with: systemctl reload caddy"
else
  echo "skipped caddy edge config; set BLUEPRINT_PIPELINE_PUBLIC_HOSTNAME to render" \
    "${REPO_ROOT}/deploy/caddy/Caddyfile"
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

systemctl daemon-reload
if [[ "${ENABLE_NOW}" == "true" ]]; then
  systemctl enable --now blueprint-pipeline-control-plane.timer
  systemctl enable --now blueprint-pubsub-handoff-listener.timer
  systemctl enable --now blueprint-capture-reconstruction-dispatcher.path
  systemctl enable --now blueprint-capture-reconstruction-dispatcher.timer
  systemctl enable --now blueprint-provider-billing-reconciler.timer
  systemctl enable --now blueprint-gpu-spend-guard.timer
  systemctl enable --now blueprint-task-evaluation-launch-reconciler.timer
  systemctl enable --now blueprint-task-evaluation-configured-controls-progression.timer
  systemctl enable --now blueprint-task-evaluation-scene-progression.timer
  systemctl enable --now blueprint-task-evaluation-configured-controls-progression.path
  systemctl enable --now blueprint-control-plane-storage-gc.timer
  systemctl enable --now blueprint-control-plane-capacity.timer
  systemctl enable --now blueprint-control-plane-preflight.timer
  systemctl enable --now blueprint-task-evaluation-launch-dispatcher.path
  systemctl enable --now blueprint-task-evaluation-launch-preparation.path
  systemctl enable --now blueprint-task-evaluation-launch-preparation.timer
  systemctl enable --now blueprint-scene-object-discovery.path
  systemctl enable --now blueprint-task-evaluation-episode-compilation.path
  systemctl enable --now blueprint-task-evaluation-episode-compilation.timer
  systemctl enable --now blueprint-task-evaluation-episode-compilation-remote.timer
  systemctl enable --now blueprint-task-evaluation-episode-compilation-remote.path
  systemctl enable --now blueprint-task-evaluation-launch-activation.path
  systemctl enable --now blueprint-task-evaluation-policy-canary-dispatcher.path
  systemctl enable --now blueprint-native-g1-team-campaign-dispatcher.timer
  systemctl enable --now blueprint-native-g1-team-campaign-settlement.timer
  systemctl enable --now blueprint-task-evaluation-terminal-resource-release.path
  systemctl enable --now blueprint-task-evaluation-launch-supervisor.timer
else
  echo "installed; enable timer with: systemctl enable --now blueprint-pipeline-control-plane.timer"
  echo "enable handoff listener with: systemctl enable --now blueprint-pubsub-handoff-listener.timer"
  echo "enable capture reconstruction queue with: systemctl enable --now blueprint-capture-reconstruction-dispatcher.path"
  echo "enable billing reconciliation with: systemctl enable --now blueprint-provider-billing-reconciler.timer"
  echo "enable spend admission guard with: systemctl enable --now blueprint-gpu-spend-guard.timer"
  echo "enable launch reconciliation with: systemctl enable --now blueprint-task-evaluation-launch-reconciler.timer"
  echo "enable configured-controls progression with: systemctl enable --now blueprint-task-evaluation-configured-controls-progression.timer"
  echo "enable compilation-result progression wake-up with: systemctl enable --now blueprint-task-evaluation-configured-controls-progression.path"
  echo "enable durable launch queue watch with: systemctl enable --now blueprint-task-evaluation-launch-dispatcher.path"
  echo "enable no-spend launch preparation queue with: systemctl enable --now blueprint-task-evaluation-launch-preparation.path"
  echo "enable bounded disk-capacity preparation retries with: systemctl enable --now blueprint-task-evaluation-launch-preparation.timer"
  echo "enable whole-splat object discovery queue with: systemctl enable --now blueprint-scene-object-discovery.path"
  echo "enable no-spend episode compilation queue with: systemctl enable --now blueprint-task-evaluation-episode-compilation.path"
  echo "enable release-window-gated launch activation queue with: systemctl enable --now blueprint-task-evaluation-launch-activation.path"
  echo "enable authority-gated paid policy canary queue with: systemctl enable --now blueprint-task-evaluation-policy-canary-dispatcher.path"
  echo "enable G1 team campaign dispatcher with: systemctl enable --now blueprint-native-g1-team-campaign-dispatcher.timer"
  echo "enable G1 team campaign settlement with: systemctl enable --now blueprint-native-g1-team-campaign-settlement.timer"
  echo "enable terminal resource release queue watch with: systemctl enable --now blueprint-task-evaluation-terminal-resource-release.path"
  echo "enable optional launch supervision with: systemctl enable --now blueprint-task-evaluation-launch-supervisor.timer"
  echo "start intake service with: systemctl enable --now blueprint-pipeline-intake.service"
fi
