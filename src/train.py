#BIBLIOTECAS
import pandas as pd
from sklearn.model_selection import train_test_split #avaliar e dividir modulos de machine learning
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from src.features import criar_atributos # chama a função do arquivo features de composição familiar
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


#IMPORTAR DADOS
dados = pd.read_csv("data/train.csv") #importa base de dados

# APLICAR DADOS À FUNÇÃO DE CRIAR ATRIBUTOS
dados = criar_atributos(dados) #aplica a base de dados à função de composição familiar

#IDENTÍFICAR CARACTERÍSTICAS NUMÉRICAS

colunas_numericas = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",

]

colunas_categoricas = [
    "Pclass",
    "Sex",
    "Embarked",
]

colunas_x = colunas_numericas + colunas_categoricas


X = dados[colunas_x] #caracteristicas ds passageiros
y = dados["Survived"] #sobreviveram ou não

#DIVISÃO ENTRE TREINO E VALIDAÇÃO

X_treino, X_validacao, y_treino, y_validacao = train_test_split( #X_validacao contém características dos passageiros que o modelo ainda não utilizará durante o treinamento
    X,
    y,
    test_size=0.2, #20% validação e 80% treino
    random_state=42,
    stratify=y #parâmetro para manter a proporção de sobreviventes e não sobreviventes
)

#CRIAR PIPELINE

tratamento_numerico = Pipeline(steps=[
    ("preencher ausentes", SimpleImputer(strategy="median")),
    ("padronizar", StandardScaler()),
]
)

tratamento_categorico = Pipeline(steps=[
    ("preencher ausentes", SimpleImputer(strategy="most_frequent")),
    ("converter_categorias", OneHotEncoder(handle_unknown="ignore")),
])

#UNIR TRATAMENTOS DE DADOS

preprocessamento = ColumnTransformer(
    transformers=[
        (
            "numericas",
            tratamento_numerico,
            colunas_numericas,
        ),
        (
            "categoricas",
            tratamento_categorico,
            colunas_categoricas,
        ),
    ]
)


print("Dados de treino:")
print("X_treino:", X_treino.shape)
print("y_treino:", y_treino.shape)

print("\nDados de validação:")
print("X_validacao:", X_validacao.shape)
print("y_validacao:", y_validacao.shape)


X_treino_preparado = preprocessamento.fit_transform(X_treino) #aprender a tratar os dados

X_validacao_preparado = preprocessamento.transform( #aplicar o tratamento aprendido
    X_validacao
)

modelo = LogisticRegression(max_iter=1000) #Criar modelo

#Treinar o modelo

modelo.fit(X_treino_preparado, y_treino)

previsoes = modelo.predict(X_validacao_preparado)


acuracia = accuracy_score(
    y_validacao,
    previsoes
)



print("\nFormato antes do tratamento:")
print(X_treino.shape)

print("\nFormato depois do tratamento:")
print(X_treino_preparado.shape)

print("\nFormato da validação preparada:")
print(X_validacao_preparado.shape)

print("\nPrimeiras previsões:")
print(previsoes[:10])

print("\nRespostas verdadeiras:")
print(y_validacao.head(10).to_numpy())

print(f"\nAcurácia: {acuracia:.2%}")

#RELATÓRIO DE CLASSIFICAÇÃO

print("\n Relatório de Classificação")

print(
    classification_report(
        y_validacao,
        previsoes,
        target_names=[
            "Não Sobreviveu",
            "Sobreviveu",
        ]
    )
)

print("\nConclusão:")
print("O modelo obteve 81,56% de acurácia.")
print("O principal ponto de melhoria é o recall dos sobreviventes: 68%.")

#MATRIZ DE CONFUSÃO

ConfusionMatrixDisplay.from_predictions(
    y_validacao,
    previsoes,
)
plt.title("Matriz de confusão")
plt.show()


# Resultado inicial:
# Acurácia: 81,56%
# Recall dos não sobreviventes: 90%
# Recall dos sobreviventes: 68%
# Principal limitação: 22 sobreviventes foram classificados incorretamente.

