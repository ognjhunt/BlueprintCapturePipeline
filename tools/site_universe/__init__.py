"""Site universe producer: a deduplicated list of US operating sites from bulk public sources.

Network access exists only in ``fetch.py``. Adapters, dedupe and the snapshot
writer are pure functions of the raw cache, so the same raw inputs give a
byte-identical snapshot.
"""

SCHEMA_VERSION = "blueprint.site_universe.snapshot.v1"
BUILDER_VERSION = "1"
