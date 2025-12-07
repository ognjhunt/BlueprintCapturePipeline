"""
Blueprint Capture Pipeline base job specifications.

This package provides lightweight, declarative stubs for each GPU-heavy
job in the pipeline (frame extraction/masking, reconstruction, mesh
extraction, object assetization, and USD authoring). The goal is to make
it easy to hydrate Cloud Run Job payloads and keep the contract between
stages explicit even before the heavy implementations are integrated.
"""

from .pipeline import build_default_pipeline

__all__ = ["build_default_pipeline"]
