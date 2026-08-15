#!/usr/bin/env python3
"""Build a current-repository Content Agents candidate bundle.

This durable operator entrypoint replaces session-scratch invocations used for
the pre-#554 candidates.  The production builder still owns all source,
candidate, route, deterministic-archive, and rehearsal validation.
"""

from blueprint_pipeline.adp_content_agents_vast import main


if __name__ == "__main__":
    raise SystemExit(main())
