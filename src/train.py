import pandas as pd #importa a biblioteca

from src.features import criar_atributos # chama a função do arquivo features de composição familiar


dados = pd.read_csv("data/train.csv") #importa base de dados

dados = criar_atributos(dados) #aplica a base de dados à função de composição familiar

colunas_x = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Pclass",
    "Sex",
    "Embarked",
]

x = dados[colunas_x] #caracteristicas ds passageiros
y = dados["Survived"] #sobreviveram ou não