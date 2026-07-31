from pathlib import Path

import joblib
import pandas as pd

from src.features import criar_atributos


caminho_modelo = Path("models/titanic_pipeline.joblib")

pipeline_modelo = joblib.load(caminho_modelo)


def prever_passageiro(dados_passageiro: dict) -> dict:
    """
    Recebe os dados de um passageiro e retorna a previsão do modelo.
    """

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
