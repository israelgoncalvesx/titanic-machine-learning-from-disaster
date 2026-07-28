from pathlib import Path
import joblib
import pandas as pd
from src.features import criar_atributos

#CARREGAR O ARQUIVO
caminho_modelo = Path("models/titanic_pipeline.joblib")

#CARREGAR PIPELINE

pipeline_modelo = joblib.load(caminho_modelo)


#PASSAGEIRO FICTÍCIO

passageiro = pd.DataFrame(
    [
        {
            "Age": 29,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 80.0,
            "Pclass": 1,
            "Sex": "female",
            "Embarked": "C",
        }
    ]
)

#ESSA FUNÇÃO ADICIONARÁ "FamilySize" e "IsAlone"
passageiro = criar_atributos(passageiro)

print("\nDados enviados ao modelo:")
print(passageiro)

previsao = pipeline_modelo.predict(passageiro)[0]

probabilidades = pipeline_modelo.predict_proba(
    passageiro
)[0]

if previsao == 1:
    resultado = "Sobreviveu"
else:
    resultado = "Não Sobreviveu"

print("\nResultado da previsao: {}".format(resultado))

print(
    f"Probabilidade de não sobreviver: "
    f"{probabilidades[0]:.2%}"
)

print(
    f"Probabilidade de sobreviver: "
    f"{probabilidades[1]:.2%}"
)


