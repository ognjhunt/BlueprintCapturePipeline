"""Minimal guarded HTTP seam; construction does not send a request.

Existing credentials must be configured outside this task. Tests inject an
in-memory opener. Every actual dispatch requires the shared paid admission grant,
the approved cumulative journal, exact endpoint, and secure-access metadata.
"""

from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import ssl
import stat
import time
import urllib.error
import urllib.request

from .adapters import MODEL, MODES, RATES, normalize, public_case, request
from .harness import Ledger, digest, exclusive, read_json, write_once

PROJECT = "proj_F2tFJuxLaovJru8RrtXRaqNj"
ENDPOINTS = {"parallel": "https://api.parallel.ai/v1/search",
             "perplexity": "https://api.perplexity.ai/search",
             "openai": "https://api.openai.com/v1/responses"}
VARIABLES = {"parallel": "PARALLEL_API_KEY", "perplexity": "PERPLEXITY_API_KEY",
             "openai": "OPENAI_API_KEY"}
MAX_RESPONSE_BYTES = 2_000_000
EXTRAS_PER_SEARCH_ATTEMPT = Decimal("0.003125")  # $0.50 / 160 attempts
SOL_PER_CALL_CEILING = Decimal("0.03548")  # all 6k input at cache-write rate + 2048 output
COUNT_ENDPOINT = "https://api.openai.com/v1/responses/input_tokens"
COUNT_METHOD = "openai.responses.input_tokens:gpt-6.1-sol:v1"
COUNT_ALLOWANCE = Decimal("0.02")  # separate allowance; endpoint fee is not documented


class LiveBlocked(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise LiveBlocked("redirect_refused_without_disclosing_credentials")


def existing_key(provider):
    """Return only in memory; no creation, grants, logging, or secret-file writes."""
    variable = VARIABLES[provider]
    key, file_name = os.environ.get(variable), os.environ.get(variable + "_FILE")
    if key and file_name:
        raise LiveBlocked("ambiguous_existing_secret_binding")
    if file_name:
        descriptor = None
        try:
            descriptor = os.open(file_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            metadata = os.fstat(descriptor)
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) & 0o077 or metadata.st_size > 65536):
                raise LiveBlocked("unsafe_existing_secret_file")
            key = os.read(descriptor, 65537).decode("utf-8").strip()
        except (OSError, UnicodeError):
            raise LiveBlocked("existing_secret_file_unavailable") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
    if not key or "\n" in key or "\r" in key:
        raise LiveBlocked("existing_secret_binding_missing_or_invalid:" + variable)
    return key


def credential_presence():
    return {provider: bool(os.environ.get(variable) or os.environ.get(variable + "_FILE"))
            for provider, variable in VARIABLES.items()}


def openai_envelope(input_items, input_token_count):
    if type(input_token_count) is not int or not 0 < input_token_count <= 6000:
        raise LiveBlocked("measured_input_token_budget_required")
    return {"method": "POST", "url": ENDPOINTS["openai"], "timeout_seconds": 30,
            "body": {"model": MODEL, "service_tier": "default", "store": False,
                     "max_output_tokens": 2048, "reasoning": {"effort": "low"},
                     "input": input_items}}


class HTTPTransport:
    def __init__(self, output, plan, access, *, opener=None, token_counter=None):
        self.output, self.plan, self.access = Path(output).resolve(), plan, access
        if (not plan.get("journal_root") or not Path(plan["journal_root"]).is_absolute()
                or self.output.resolve() != Path(plan["journal_root"]).resolve()):
            raise LiveBlocked("canonical_experiment_journal_root_required")
        if (plan.get("execution") != "live_authorized" or plan.get("budget_usd") != "10.00"
                or plan.get("model") != MODEL or plan.get("openai_project") != PROJECT
                or plan.get("case_count") != 20 or not plan.get("public_inputs_sha256")
                or not plan.get("reviewer_spec_sha256")
                or set(plan.get("case_hashes", {})) != {f"{i:02d}" for i in range(1, 21)}):
            raise LiveBlocked("exact_live_plan_missing")
        # Caller supplies a receipt for owner-configured existing access, never a
        # secret. Merely discovering a connector or approving spend is insufficient.
        if (access.get("status") != "existing_access_configured"
                or access.get("openai_project") != PROJECT
                or set(access.get("allowed_hosts", [])) != {
                    "api.parallel.ai", "api.perplexity.ai", "api.openai.com"}
                or not (access.get("parallel_x_api_key_supported") is True
                        or (access.get("parallel_x_api_key_supported") == "unverified"
                            and access.get("parallel_header_pilot_probe_authorized") is True))):
            raise LiveBlocked("secure_existing_access_or_proxy_header_support_unconfirmed")
        self.opener = opener
        self.token_counter = token_counter

    def model_input(self, cell, role, public_input, reviewer_oracle_bytes=None):
        """Compose exactly admitted public inputs and verified retained evidence."""
        public_case(public_input)
        if digest(public_input) != self.plan["case_hashes"].get(cell.split("_", 1)[0]):
            raise LiveBlocked("model_public_input_hash_mismatch")
        content = {"role": role, "public_case": public_input,
                   "instruction": "Preserve unsupported claims as unknown; no site qualification."}
        if role in ("synthesis", "reviewer"):
            ledger = Ledger(self.output / "live_journal.jsonl", "10.00")
            mode = cell.split("_", 1)[1]

            def retained(provider, retained_role):
                reservations = [event for event in ledger.events if event["kind"] == "reserved"
                                and event.get("cell") == cell and event.get("provider") == provider
                                and event.get("role") == retained_role
                                and ledger.states[event["attempt_id"]] == "completed"]
                if len(reservations) != 1:
                    raise LiveBlocked("exact_retained_role_evidence_required")
                key = reservations[0]["attempt_id"]
                result = read_json(self.output / "live_raw" / (key + ".json"))
                completed = next(event for event in ledger.events
                                 if event["attempt_id"] == key and event["kind"] == "completed")
                if digest(result) != completed["raw_sha256"]:
                    raise LiveBlocked("model_source_digest_failure")
                return result["raw"]

            content["sources"] = normalize(mode, retained(mode.split("_")[0], "search"))
            if role == "reviewer":
                if (not isinstance(reviewer_oracle_bytes, bytes)
                        or hashlib.sha256(reviewer_oracle_bytes).hexdigest()
                        != self.plan.get("reviewer_oracle_sha256")):
                    raise LiveBlocked("isolated_reviewer_exact_oracle_required")
                content["answer"] = retained("openai", "synthesis")
                content["oracle"] = json.loads(reviewer_oracle_bytes)
                content["rubric_sha256"] = self.plan["reviewer_spec_sha256"]
        return json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def send(self, provider, envelope, *, cell, role, attempt, grant,
             input_token_count=None, public_input=None, reviewer_oracle_bytes=None):
        counting = provider == "openai" and role in ("count_synthesis", "count_reviewer")
        expected_url = COUNT_ENDPOINT if counting else ENDPOINTS.get(provider)
        if provider not in ENDPOINTS or envelope.get("url") != expected_url:
            raise LiveBlocked("endpoint_not_allowed")
        if envelope.get("method") != "POST" or envelope.get("timeout_seconds") != 30:
            raise LiveBlocked("method_or_timeout_not_allowed")
        body = envelope["body"]
        if provider == "openai":
            input_role = role.removeprefix("count_") if counting else role
            if public_input is None or body.get("input") != self.model_input(
                    cell, input_role, public_input, reviewer_oracle_bytes):
                raise LiveBlocked("model_input_provenance_mismatch")
            if counting:
                if (set(body) != {"model", "input", "reasoning"} or body["model"] != MODEL
                        or body["reasoning"] != {"effort": "low"} or attempt != 1
                        or self.plan.get("tokenizer_id") != COUNT_METHOD
                        or self.access.get("tokenizer_id") != COUNT_METHOD):
                    raise LiveBlocked("exact_provider_count_request_required")
                resource_class, reserve = "openai_api_candidate", COUNT_ALLOWANCE
            else:
                self._validate_inference(body, input_token_count, role, attempt)
                resource_class, reserve = "openai_api_candidate", SOL_PER_CALL_CEILING
        else:
            mode = provider + "_" + (body.get("mode", "") if provider == "parallel"
                                     else "fast" if body.get("search_type") == "fast"
                                     else "standard" if body.get("search_type") == "web" else "")
            if mode not in RATES or role != "search" or attempt not in (1, 2):
                raise LiveBlocked("raw_search_mode_or_attempt_not_allowed")
            if public_input is None:
                raise LiveBlocked("frozen_public_input_required")
            public_case(public_input)
            if (digest(public_input) != self.plan["case_hashes"].get(cell.split("_", 1)[0])
                    or envelope != request(mode, public_input, 0)):
                raise LiveBlocked("public_input_or_request_hash_mismatch")
            resource_class, reserve = "evaluator_api", RATES[mode] + EXTRAS_PER_SEARCH_ATTEMPT
        return self._dispatch(provider, envelope, cell=cell, role=role, attempt=attempt,
                              grant=grant, resource_class=resource_class, reserve=reserve,
                              mode=None if provider == "openai" else mode, counting=counting)

    def _validate_inference(self, body, input_token_count, role, attempt):
        if (not callable(self.token_counter)
                or self.access.get("tokenizer_id") != self.plan.get("tokenizer_id")
                or not self.plan.get("tokenizer_id")
                or self.token_counter(body.get("input")) != input_token_count):
            raise LiveBlocked("pinned_model_tokenizer_receipt_required")
        if (body.get("model") != MODEL or body.get("max_output_tokens") != 2048
                    or body.get("service_tier") != "default" or body.get("store") is not False
                    or set(body) != {"model", "max_output_tokens", "service_tier", "store",
                                    "input", "reasoning"}
                    or body.get("reasoning") != {"effort": "low"}
                    or type(input_token_count) is not int or not 0 < input_token_count <= 6000):
            raise LiveBlocked("model_tools_tier_or_token_budget_not_allowed")
        if role not in ("planner", "synthesis", "reviewer") or attempt != 1:
            raise LiveBlocked("inference_role_or_retry_not_allowed")

    def _dispatch(self, provider, envelope, *, cell, role, attempt, grant,
                  resource_class, reserve, mode, counting):
        from blueprint_pipeline.paid_resource_admission import require_paid_resource_admission_grant

        body = envelope["body"]
        binding = digest(self.plan)
        require_paid_resource_admission_grant(grant, resource_class=resource_class,
                                             allocation_binding_digest=binding,
                                             require_allocation_binding=True)
        cells = {f"{index:02d}_{mode}" for index in range(1, 21) for mode in MODES}
        case_number = int(cell.split("_", 1)[0]) if cell in cells else 0
        if (cell not in cells or (provider != "openai" and not cell.endswith(mode))
                or self.plan.get("phase") not in ("pilot", "remaining")
                or (self.plan["phase"] == "pilot") != (case_number <= 2)):
            raise LiveBlocked("case_or_phase_not_allowed")
        key = digest({"plan": binding, "provider": provider, "cell": cell, "role": role,
                      "attempt": attempt, "request": envelope})
        key_value = existing_key(provider)
        headers = {"Content-Type": "application/json"}
        if provider == "parallel":
            headers["x-api-key"] = key_value
        else:
            headers["Authorization"] = "Bearer " + key_value
        if provider == "openai":
            headers["OpenAI-Project"] = PROJECT
        with exclusive(self.output):
            # Phase is not a new spend account: retain the same experiment scope
            # while pilot and remaining phases use one cumulative journal.
            scope = {key: value for key, value in self.plan.items() if key != "phase"}
            write_once(self.output / "live_scope.json", scope)
            ledger = Ledger(self.output / "live_journal.jsonl", "10.00")
            if key in ledger.states:
                if ledger.states[key] == "completed":
                    retained = read_json(self.output / "live_raw" / (key + ".json"))
                    completed = next(event for event in ledger.events
                                     if event["attempt_id"] == key and event["kind"] == "completed")
                    if digest(retained) != completed["raw_sha256"]:
                        raise LiveBlocked("retained_live_response_integrity_failure")
                    return retained
                raise LiveBlocked("uncertain_existing_attempt_requires_reconciliation_no_resend")
            related = [event for event in ledger.events if event["kind"] == "reserved"
                       and event.get("cell") == cell and event.get("provider") == provider
                       and event.get("role") == role]
            if related:
                if (provider == "openai" or attempt != 2 or len(related) != 1
                        or ledger.states[related[0]["attempt_id"]] != "not_accepted"):
                    raise LiveBlocked("retry_without_proven_nonacceptance")
            elif attempt != 1:
                raise LiveBlocked("initial_attempt_required")
            if case_number <= 2 and attempt != 1:
                raise LiveBlocked("pilot_exactly_eight_provider_attempts_no_retry")
            pilot_exposure = sum((ledger.reservations[event["attempt_id"]]
                                  for event in ledger.events if event["kind"] == "reserved"
                                  and int(event["cell"].split("_", 1)[0]) <= 2
                                  and ledger.states[event["attempt_id"]] != "not_accepted"),
                                 Decimal(0))
            if case_number <= 2 and pilot_exposure + reserve > 1:
                raise LiveBlocked("pilot_inclusive_cap_before_dispatch")
            ledger.append("reserved", key, amount_usd=str(reserve), cell=cell,
                          provider=provider, role=role, request_sha256=digest(envelope))
            try:
                started = time.monotonic()
                opener = self.opener
                if opener is None:
                    context = ssl.create_default_context(cafile=os.environ.get("CODEX_PROXY_CERT"))
                    opener = urllib.request.build_opener(NoRedirect(),
                                                        urllib.request.HTTPSHandler(context=context))
                req = urllib.request.Request(envelope["url"], data=json.dumps(body).encode(),
                                             headers=headers, method="POST")
                with opener.open(req, timeout=30) as response:
                    if response.status != 200:
                        raise LiveBlocked("non_success_response")
                    blob = response.read(MAX_RESPONSE_BYTES + 1)
                if len(blob) > MAX_RESPONSE_BYTES:
                    raise LiveBlocked("response_size_cap")
                raw = json.loads(blob)
                retained = {"request_sha256": digest(envelope), "raw": raw,
                            "latency_seconds": time.monotonic() - started,
                            "actual_billing": "unreconciled; full reservation retained"}
                write_once(self.output / "live_raw" / (key + ".json"), retained)
                if provider != "openai":
                    normalize(mode, raw)
                elif counting:
                    if (raw.get("object") != "response.input_tokens"
                            or type(raw.get("input_tokens")) is not int
                            or not 0 < raw["input_tokens"] <= 6000):
                        raise LiveBlocked("provider_count_failed_or_input_budget_exceeded")
                elif raw.get("model") != MODEL:
                    raise LiveBlocked("returned_model_identity_mismatch")
                ledger.append("completed", key, raw_sha256=digest(retained))
                return retained
            except Exception:
                # Includes HTTP errors, disconnects, redirects and parse errors.
                # Do not echo request/headers/response/error text or free exposure.
                ledger.append("uncertain", key, reason="submission_or_response_uncertain")
                raise LiveBlocked("uncertain_submission_full_reservation_no_automatic_retry") from None
