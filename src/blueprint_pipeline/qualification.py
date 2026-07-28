"""Deprecated import alias for :mod:`blueprint_pipeline.site_package_orchestrator`.

The capture->site-package orchestration spine was historically named
``qualification.py`` for its qualification-era origin. Platform doctrine treats
qualification/readiness *outputs* as optional support artifacts, while this
module is the core capture->package orchestrator — the old name misled agents
and engineers about where the product core lives, so the module was renamed
(see ``docs/architecture/refactor-hotspots.md`` and ``AGENTS.md``).

Import ``blueprint_pipeline.site_package_orchestrator`` directly in new code.
This alias keeps older imports and dotted monkeypatch paths (for example
``blueprint_pipeline.qualification.run_privacy_postprocess``) working by
aliasing this module name to the renamed module object itself.
"""

import sys

from blueprint_pipeline import site_package_orchestrator as _site_package_orchestrator

sys.modules[__name__] = _site_package_orchestrator
