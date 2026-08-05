"""Entrada compatível para executar a aplicação Streamlit principal.

O arquivo oficial do deploy é ``app/streamlit_app.py``. Este módulo existe
para que o comando antigo ``streamlit run app/main.py`` continue funcionando.
"""

import sys
from pathlib import Path


CAMINHO_PROJETO = Path(__file__).resolve().parents[1]

if str(CAMINHO_PROJETO) not in sys.path:
    sys.path.insert(0, str(CAMINHO_PROJETO))

from app.streamlit_app import *  # noqa: E402,F403
