"""AetherScientist dual entrypoint supporting Streamlit and Hugging Face Gradio."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import streamlit as st  # type: ignore

    _IS_STREAMLIT = st.runtime.exists()
except Exception:
    _IS_STREAMLIT = False

if _IS_STREAMLIT:
    from ui.app import main  # noqa: E402

    if __name__ == "__main__":
        main()
else:
    from ui.gradio_app import create_gradio_app  # noqa: E402

    if __name__ == "__main__":
        app = create_gradio_app()
        app.launch()
