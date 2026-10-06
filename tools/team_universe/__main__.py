"""``python -m tools.team_universe <command>``: the owner command (see ``cli.py``)."""
import sys

from tools.team_universe.cli import run

sys.exit(run())
