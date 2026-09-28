"""Dispatch the trusted proxy or the explicit command in this synthetic image."""
import os
import sys
args = sys.argv[1:]
if args and args[0] == "serve":
    args = ["python", "-m", "blueprint_pipeline.company_policy_proxy", *args]
os.execvp(args[0], args)
