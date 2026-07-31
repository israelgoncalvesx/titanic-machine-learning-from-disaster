from typing import Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from src.predict import prever_passageiro


class PassageiroEntrada(BaseModel):
    """Dados necessários para realizar uma previsão."""

    model_config = ConfigDict(extra="forbid")

    Age: float = Field(
        ge=0,
        le=120,
        description="Idade do passageiro.",
        examples=[29],
    )
    SibSp: int = Field(
        ge=0,
        description="Quantidade de irmãos, irmãs ou cônjuges a bordo.",
        examples=[0],
    )
    Parch: int = Field(
        ge=0,
        description="Quantidade de pais ou filhos a bordo.",
        examples=[0],
    )
    Fare: float = Field(
        ge=0,
        description="Valor pago pela passagem.",
        examples=[80.0],
    )
    Pclass: Literal[1, 2, 3] = Field(
        description="Classe da passagem.",
        examples=[1],
    )
    Sex: Literal["male", "female"] = Field(
        description="Sexo registrado no conjunto de dados.",
        examples=["female"],
    )
    Embarked: Literal["S", "C", "Q"] = Field(
        description="Porto de embarque.",
        examples=["C"],
    )


class PrevisaoSaida(BaseModel):
    """Resposta devolvida pelo modelo."""

    classe: Literal[0, 1]
    resultado: str
    probabilidade_nao_sobreviver: float = Field(
        ge=0,
        le=1,
    )
    probabilidade_sobreviver: float = Field(
        ge=0,
        le=1,
    )


app = FastAPI(
    title="Titanic Survival Prediction API",
    description=(
        "API para estimar a probabilidade de sobrevivência "
        "de um passageiro do Titanic usando o pipeline treinado."
    ),
    version="1.0.0",
)


@app.get(
    "/",
    tags=["Status"],
)
def inicio() -> dict[str, str]:
    """Apresenta informações básicas sobre a API."""
    return {
        "mensagem": "Titanic Survival Prediction API",
        "documentacao": "/docs",
        "status": "online",
    }


@app.get(
    "/health",
    tags=["Status"],
)
def verificar_saude() -> dict[str, str]:
    """Informa se o processo da API está respondendo."""
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PrevisaoSaida,
    status_code=status.HTTP_200_OK,
    tags=["Previsão"],
)
def realizar_previsao(
    passageiro: PassageiroEntrada,
) -> dict:
    """Valida os dados recebidos e retorna a previsão do modelo."""
    try:
        return prever_passageiro(
            passageiro.model_dump()
        )
    except (TypeError, ValueError) as erro:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(erro),
        ) from erro
    except (FileNotFoundError, OSError) as erro:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="O modelo treinado não está disponível.",
        ) from erro
