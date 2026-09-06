"""Streamlit entrypoint wrapper for Cloud deployment."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.app import main  # noqa: E402

if __name__ == "__main__":
    main()
