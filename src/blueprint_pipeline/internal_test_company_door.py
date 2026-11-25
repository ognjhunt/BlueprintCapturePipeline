"""Act for Blueprint's internal test company, for the operator door's ``test-company`` request.

The website keeps one internal test company (Blueprint-WebApp ``server/utils/internalTestCompany.ts``, owner
2026-11-25): it sees Blueprint's internal test jobs only, its emails go only to the inbox the website's own setting
names, and its introductions are test events, never billable. This module only asks the website, over the same signed
Pipeline -> WebApp transport as ``internal_test_job_door``, to do one thing, and returns the website's answer:

- ``ensure``: the one test company record (``POST /api/internal/pipeline/test-company``);
- ``send``: one test job sent to it, as a site's job is sent to every approved company (``.../test-company/send``);
- ``answer``: its answer on one test job, through the website's own answer path (``.../test-company/answer``);
- ``accept``: the site's Accept for it on one test job, through the site's own introduction code, so no site token
  leaves the website (``.../test-company/accept``);
- ``site-link``: one test job's site owner link (``.../test-jobs/<job>/site-link``);
- ``page``: its own page of one test job, read as its page read reads it, answered as a redacted summary with the job
  summary PDF's text and no address, ticket, signature or token (``.../test-company/page``).

Nothing here deletes, changes a limit or spends by itself. Each call runs the website's own path, which may start the
model passes that path starts for any job (the screening of the words read across, on a send, an answer or a page read;
a page read also the job's brief when it has none current), each held by the website against the job's one USD 40 cap.
The website refuses every job that is not an internal test job, and every company but the test company; this module
names no email address.
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .internal_test_job_door import Refused, post_signed

KIND = "test-company"
#: The website's answers a company gives (Blueprint-WebApp ``client/src/lib/companyResponses.ts``).
STATES = ("interested", "declined", "needs_something", "tested_can_do", "tested_cannot_yet")
#: The website's id for a test job (``server/routes/internal-test-jobs.ts`` ``internalTestJobId``).
JOB_ID = re.compile(r"itest-[0-9a-f]{40}")
_WEBSITE_REFUSAL = re.compile(r"test_company_webapp_http_4[0-9]{2}(:.*)?")


def validated(request: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """The website path and body for the door's request, checked again here as the door checked it."""
    if not isinstance(request, Mapping) or request.get("kind") != KIND:
        raise Refused("test_company_request_invalid")
    action, job_id = request.get("action"), request.get("job_id")
    keys = {"ensure": {"kind", "action"}, "send": {"kind", "action", "job_id"},
            "site-link": {"kind", "action", "job_id"}, "accept": {"kind", "action", "job_id"},
            "page": {"kind", "action", "job_id"}, "answer": {"kind", "action", "job_id", "state", "inputs"}}
    if action not in keys or not set(request) <= keys[action] or (action != "answer" and set(request) != keys[action]):
        raise Refused("test_company_request_invalid")
    if action == "ensure":
        return "/api/internal/pipeline/test-company", {}
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise Refused("test_company_job_invalid")
    if action == "send":
        return "/api/internal/pipeline/test-company/send", {"job_id": job_id}
    if action == "site-link":
        return f"/api/internal/pipeline/test-jobs/{job_id}/site-link", {}
    if action == "accept":
        return "/api/internal/pipeline/test-company/accept", {"job_id": job_id}
    if action == "page":
        return "/api/internal/pipeline/test-company/page", {"job_id": job_id}
    state, inputs = request.get("state"), request.get("inputs", "")
    if state not in STATES:
        raise Refused("test_company_state_invalid")
    if not isinstance(inputs, str) or len(inputs) > 2000 or not all(ch.isprintable() or ch == "\n" for ch in inputs):
        raise Refused("test_company_inputs_invalid")
    return "/api/internal/pipeline/test-company/answer", {"job_id": job_id, "state": state, "inputs": inputs}


def run(request: Mapping[str, Any], *, post: Callable[..., dict[str, Any]] = post_signed) -> dict[str, Any]:
    path, body = validated(request)
    try:
        answer = post(path, body, prefix="test_company")
    except ValueError as exc:
        if _WEBSITE_REFUSAL.fullmatch(str(exc)):
            raise Refused(str(exc)) from None  # the website said no: nothing was done
        raise
    return {"door_outcome": "done", "action": request["action"], "website": answer}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--request-json", required=True)
    args = parser.parse_args(argv)
    try:
        report, code = run(json.loads(args.request_json)), 0
    except Refused as exc:
        report, code = {"door_outcome": "refused", "code": str(exc)}, 2
    except (ValueError, KeyError, TypeError, OSError) as exc:
        text = str(exc)
        report = {"door_outcome": "failed", "code": text if re.fullmatch(r"[a-z0-9_:.-]{1,200}", text)
                  else type(exc).__name__}
        code = 1
    print(json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
