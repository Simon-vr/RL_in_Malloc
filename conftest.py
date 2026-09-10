"""Root conftest: make the repository importable when running pytest."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
