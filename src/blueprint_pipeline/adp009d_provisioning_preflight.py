"""Report every missing provisioning prerequisite at once, not one per run.

Three consecutive paid runs each discovered exactly one missing dependency --
ensurepip, then linux/input.h, then Python.h -- because the script failed at the
first thing it needed and stopped.  Each discovery cost a GPU run and about
eleven minutes to learn one fact that a few seconds of checking would have
produced alongside all the others.

So this checks everything first and reports the complete set.  It deliberately
does not stop at the first failure, and it deliberately runs before any install,
because the value is entirely in learning all of it at once.

The checks are the ones live runs actually failed on, plus the ones the next
stages need.  Nothing here is speculative: every entry either broke a run or is
required by a command the script already issues.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = "adp009d_provisioning_preflight.v1"

# Commands the provisioning script issues directly.
REQUIRED_COMMANDS = ("curl", "git", "apt-get", "gcc", "cc")
# Headers whose absence broke a live run.  Python.h's directory varies by
# version, so it is probed rather than assumed at one path.
REQUIRED_HEADERS = ("linux/input.h",)
# Interpreters: the venv is built from the system one, never from Isaac's.
REQUIRED_INTERPRETERS = ("/usr/bin/python3",)
HEADER_SEARCH_ROOTS = ("/usr/include", "/usr/local/include")


def _command_present(name: str) -> bool:
    return shutil.which(name) is not None


def _header_present(relative: str) -> bool:
    return any(Path(root, relative).is_file() for root in HEADER_SEARCH_ROOTS)


def _python_header_present(interpreter: str) -> bool:
    """Python.h lives under a version- and platform-specific include directory."""

    try:
        completed = subprocess.run(
            [
                interpreter,
                "-c",
                "import sysconfig;print(sysconfig.get_paths()['include'])",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    include = completed.stdout.strip()
    return bool(include) and Path(include, "Python.h").is_file()


def collect_preflight() -> dict:
    """Check every prerequisite and report the complete set of failures."""

    missing_commands = [name for name in REQUIRED_COMMANDS if not _command_present(name)]
    # cc and gcc are interchangeable; only both being absent is a failure.
    if "gcc" in missing_commands and "cc" in missing_commands:
        compiler_missing = True
    else:
        compiler_missing = False
        missing_commands = [n for n in missing_commands if n not in ("gcc", "cc")]

    missing_headers = [name for name in REQUIRED_HEADERS if not _header_present(name)]
    missing_interpreters = [
        path for path in REQUIRED_INTERPRETERS if not Path(path).is_file()
    ]

    python_header_ok = any(
        _python_header_present(path)
        for path in REQUIRED_INTERPRETERS
        if Path(path).is_file()
    )
    if not python_header_ok:
        missing_headers.append("Python.h")

    blockers: list[str] = []
    blockers.extend(f"missing_command:{name}" for name in missing_commands)
    if compiler_missing:
        blockers.append("missing_command:c_compiler")
    blockers.extend(f"missing_header:{name}" for name in missing_headers)
    blockers.extend(f"missing_interpreter:{path}" for path in missing_interpreters)

    return {
        "schema_version": SCHEMA_VERSION,
        # "ready" means nothing is known to be missing.  It is not a promise the
        # install will succeed, only that these specific prerequisites are met.
        "status": "ready" if not blockers else "incomplete",
        "blockers": sorted(blockers),
        "checked_commands": list(REQUIRED_COMMANDS),
        "checked_headers": [*REQUIRED_HEADERS, "Python.h"],
        "checked_interpreters": list(REQUIRED_INTERPRETERS),
        "reports_all_failures_not_the_first": True,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    report = collect_preflight()
    if arguments:
        output = Path(arguments[0])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for blocker in report["blockers"]:
        print(f"BLUEPRINT_ADP009D_PREFLIGHT_MISSING:{blocker}")
    print(f"BLUEPRINT_ADP009D_PREFLIGHT:{report['status']}")
    # Never fatal: the point is to report everything, and the install that
    # follows installs most of what this finds missing.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
