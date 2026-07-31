from pathlib import Path

import joblib
import pandas as pd

from src.features import criar_atributos


caminho_modelo = Path("models/titanic_pipeline.joblib")

pipeline_modelo = joblib.load(caminho_modelo)

COLUNAS_OBRIGATORIAS = {
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Pclass",
    "Sex",
    "Embarked",
}


def prever_passageiro(dados_passageiro: dict) -> dict:
    """
    Recebe os dados de um passageiro e retorna a previsão do modelo.
    """

    if not isinstance(dados_passageiro, dict):
        raise TypeError(
            "Os dados do passageiro devem ser enviados em um dicionário."
        )

    campos_ausentes = (
            COLUNAS_OBRIGATORIAS #compara os atributos dos dicionários com campos que deveriam existir com os que realmente existem
            - dados_passageiro.keys()
    )

    if campos_ausentes: #verifica campos ausentes
        raise ValueError(
            f"Campos obrigatórios ausentes: "
            f"{sorted(campos_ausentes)}"
        )

    if dados_passageiro["Pclass"] not in {1, 2, 3}: #verifica se a classe informada existe
        raise ValueError(
            "Pclass deve ser 1, 2 ou 3."
        )

    if dados_passageiro["Sex"] not in {"male", "female"}: #verifica o gênero
        raise ValueError(
            "Sex deve ser 'male' ou 'female'."
        )

    if dados_passageiro["Embarked"] not in {"S", "C", "Q"}: #verifica onde o passageiro embarcou
        raise ValueError(
            "Embarked deve ser 'S', 'C' ou 'Q'."
        )

    idade = dados_passageiro["Age"]

    if not isinstance(idade, (int, float)): #verifica se a idade é um número
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

    previsao = pipeline_modelo.predict(passageiro)[0]

    probabilidades = pipeline_modelo.predict_proba(
        passageiro
    )[0]

    if previsao == 1:
        resultado = "Sobreviveu"
    else:
        resultado = "Não sobreviveu"

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
        resultado_previsao["resultado"]
    )

    print(
        "Probabilidade de não sobreviver:",
        f"{resultado_previsao['probabilidade_nao_sobreviver']:.2%}"
    )

    print(
        "Probabilidade de sobreviver:",
        f"{resultado_previsao['probabilidade_sobreviver']:.2%}"
    )
