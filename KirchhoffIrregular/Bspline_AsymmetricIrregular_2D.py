"""Boundary-fitted B-spline solver for the asymmetric irregular plate.

The implementation lives in ``BoundaryFitted_Bspline_2D.py``.  This public
entry point is retained so existing commands and imports continue to work.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KirchhoffIrregular.BoundaryFitted_Bspline_2D import *  # noqa: F401,F403


if __name__ == "__main__":
    run_cli()
