import pandas as pd


def criar_atributos(dados: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variáveis relacionadas à composição familiar dos passageiros.

    Parâmetros
    ----------
    dados:
        DataFrame contendo as colunas SibSp e Parch.

    Retorno
    -------
    pd.DataFrame
        Cópia dos dados com as colunas FamilySize e IsAlone.
    """
    dados_transformados = dados.copy()

    dados_transformados["FamilySize"] = (
        dados_transformados["SibSp"]
        + dados_transformados["Parch"]
        + 1
    )

    dados_transformados["IsAlone"] = (
        dados_transformados["FamilySize"] == 1
    ).astype(int)

    return dados_transformados