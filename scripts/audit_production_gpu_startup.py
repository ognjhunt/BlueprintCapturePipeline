#!/usr/bin/env python3
"""CLI wrapper for the production GPU startup promotion gate."""

from blueprint_pipeline.production_gpu_worker_pool import readiness_main


if __name__ == "__main__":
    raise SystemExit(readiness_main())
