from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features import criar_atributos


CAMINHO_PROJETO = Path(__file__).resolve().parents[1]
CAMINHO_MODELO = (
    CAMINHO_PROJETO
    / "models"
    / "titanic_pipeline.joblib"
)

COLUNAS_OBRIGATORIAS = {
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Pclass",
    "Sex",
    "Embarked",
}


@lru_cache(maxsize=1)
def carregar_modelo():
    """Carrega o pipeline treinado uma única vez."""
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            "O modelo treinado não foi encontrado em "
            f"{CAMINHO_MODELO}."
        )

    return joblib.load(CAMINHO_MODELO)


def prever_passageiro(dados_passageiro: dict) -> dict:
    """Recebe os dados de um passageiro e retorna a previsão."""
    if not isinstance(dados_passageiro, dict):
        raise TypeError(
            "Os dados do passageiro devem ser enviados em um dicionário."
        )

    campos_ausentes = (
        COLUNAS_OBRIGATORIAS
        - dados_passageiro.keys()
    )

    if campos_ausentes:
        raise ValueError(
            "Campos obrigatórios ausentes: "
            f"{sorted(campos_ausentes)}"
        )

    if dados_passageiro["Pclass"] not in {1, 2, 3}:
        raise ValueError(
            "Pclass deve ser 1, 2 ou 3."
        )

    if dados_passageiro["Sex"] not in {"male", "female"}:
        raise ValueError(
            "Sex deve ser 'male' ou 'female'."
        )

    if dados_passageiro["Embarked"] not in {"S", "C", "Q"}:
        raise ValueError(
            "Embarked deve ser 'S', 'C' ou 'Q'."
        )

    idade = dados_passageiro["Age"]

    if not isinstance(idade, (int, float)):
        raise TypeError(
            "Age deve ser um número."
        )

    if idade < 0 or idade > 120:
        raise ValueError(
            "Age deve estar entre 0 e 120."
        )

    quantidade_irmaos_conjuges = dados_passageiro["SibSp"]

    if not isinstance(quantidade_irmaos_conjuges, int):
        raise TypeError(
            "SibSp deve ser um número inteiro."
        )

    if quantidade_irmaos_conjuges < 0:
        raise ValueError(
            "SibSp não pode ser negativo."
        )

    quantidade_pais_filhos = dados_passageiro["Parch"]

    if not isinstance(quantidade_pais_filhos, int):
        raise TypeError(
            "Parch deve ser um número inteiro."
        )

    if quantidade_pais_filhos < 0:
        raise ValueError(
            "Parch não pode ser negativo."
        )

    tarifa = dados_passageiro["Fare"]

    if not isinstance(tarifa, (int, float)):
        raise TypeError(
            "Fare deve ser um número."
        )

    if tarifa < 0:
        raise ValueError(
            "Fare não pode ser negativa."
        )

    passageiro = pd.DataFrame([dados_passageiro])
    passageiro = criar_atributos(passageiro)

    pipeline_modelo = carregar_modelo()

    previsao = pipeline_modelo.predict(
        passageiro
    )[0]

    probabilidades = pipeline_modelo.predict_proba(
        passageiro
    )[0]

    resultado = (
        "Sobreviveu"
        if previsao == 1
        else "Não sobreviveu"
    )

    return {
        "classe": int(previsao),
        "resultado": resultado,
        "probabilidade_nao_sobreviver": float(
            probabilidades[0]
        ),
        "probabilidade_sobreviver": float(
            probabilidades[1]
        ),
    }


if __name__ == "__main__":
    passageiro_exemplo = {
        "Age": 29,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80.0,
        "Pclass": 1,
        "Sex": "female",
        "Embarked": "C",
    }

    resultado_previsao = prever_passageiro(
        passageiro_exemplo
    )

    print(
        "\nResultado:",
        resultado_previsao["resultado"],
    )

    print(
        "Probabilidade de não sobreviver:",
        f"{resultado_previsao['probabilidade_nao_sobreviver']:.2%}",
    )

    print(
        "Probabilidade de sobreviver:",
        f"{resultado_previsao['probabilidade_sobreviver']:.2%}",
    )
