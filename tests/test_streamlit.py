from streamlit.testing.v1 import AppTest


TIMEOUT_SEGUNDOS = 30


def executar_app(caminho: str) -> AppTest:
    app = AppTest.from_file(caminho)
    app.run(timeout=TIMEOUT_SEGUNDOS)

    assert not app.exception

    return app


def test_pagina_principal_abre_e_realiza_previsao():
    app = executar_app("app/streamlit_app.py")

    app.button[0].click().run(timeout=TIMEOUT_SEGUNDOS)

    assert not app.exception
    assert len(app.success) == 1
    assert len(app.metric) == 2


def test_entrada_legada_abre_a_pagina_principal():
    app = executar_app("app/main.py")

    assert app.title[0].value == "Previsão de Sobrevivência no Titanic"


def test_dashboard_abre_com_indicadores():
    app = executar_app("app/pages/1_Dashboard.py")

    assert app.title[0].value == "Dashboard do Titanic"
    assert len(app.metric) == 4
