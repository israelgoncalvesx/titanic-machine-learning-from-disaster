import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


PASSAGEIRO_VALIDO = {
    "Age": 29,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 80.0,
    "Pclass": 1,
    "Sex": "female",
    "Embarked": "C",
}


def test_rota_inicial_retorna_status_online():
    resposta = client.get("/")

    assert resposta.status_code == 200
    assert resposta.json()["status"] == "online"
    assert resposta.json()["documentacao"] == "/docs"


def test_rota_health_retorna_ok():
    resposta = client.get("/health")

    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_previsao_valida_retorna_probabilidades():
    resposta = client.post(
        "/predict",
        json=PASSAGEIRO_VALIDO,
    )

    assert resposta.status_code == 200

    resultado = resposta.json()

    assert resultado["classe"] in {0, 1}
    assert resultado["resultado"] in {
        "Sobreviveu",
        "Não sobreviveu",
    }
    assert 0 <= resultado["probabilidade_sobreviver"] <= 1
    assert 0 <= resultado["probabilidade_nao_sobreviver"] <= 1
    assert (
        resultado["probabilidade_sobreviver"]
        + resultado["probabilidade_nao_sobreviver"]
    ) == pytest.approx(1.0)


def test_previsao_rejeita_idade_negativa():
    passageiro_invalido = {
        **PASSAGEIRO_VALIDO,
        "Age": -1,
    }

    resposta = client.post(
        "/predict",
        json=passageiro_invalido,
    )

    assert resposta.status_code == 422


def test_previsao_rejeita_classe_invalida():
    passageiro_invalido = {
        **PASSAGEIRO_VALIDO,
        "Pclass": 4,
    }

    resposta = client.post(
        "/predict",
        json=passageiro_invalido,
    )

    assert resposta.status_code == 422


def test_previsao_rejeita_campo_extra():
    passageiro_invalido = {
        **PASSAGEIRO_VALIDO,
        "Name": "Passageiro de teste",
    }

    resposta = client.post(
        "/predict",
        json=passageiro_invalido,
    )

    assert resposta.status_code == 422
