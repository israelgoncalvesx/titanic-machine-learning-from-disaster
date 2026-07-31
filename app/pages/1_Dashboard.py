from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Dashboard Titanic",
    page_icon="📊",
    layout="wide",
)


CAMINHO_PROJETO = Path(__file__).resolve().parents[2]
CAMINHO_DADOS_LOCAL = CAMINHO_PROJETO / "data" / "train.csv"

URL_DADOS = (
    "https://raw.githubusercontent.com/"
    "Bhasfe/titanic/master/train.csv"
)

COLUNAS_OBRIGATORIAS = {
    "PassengerId",
    "Survived",
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
}

ROTULOS_SEXO = {
    "female": "Feminino",
    "male": "Masculino",
}

ROTULOS_CLASSE = {
    1: "1ª classe",
    2: "2ª classe",
    3: "3ª classe",
}

ROTULOS_EMBARQUE = {
    "S": "Southampton",
    "C": "Cherbourg",
    "Q": "Queenstown",
}

ROTULOS_VARIAVEIS = {
    "Sex": "Sexo",
    "Pclass": "Classe da passagem",
    "Embarked": "Porto de embarque",
}


@st.cache_data
def carregar_dados() -> pd.DataFrame:
    """Carrega o CSV local ou usa uma cópia pública."""
    caminho = (
        CAMINHO_DADOS_LOCAL
        if CAMINHO_DADOS_LOCAL.exists()
        else URL_DADOS
    )

    dados_carregados = pd.read_csv(caminho)

    colunas_ausentes = (
        COLUNAS_OBRIGATORIAS
        - set(dados_carregados.columns)
    )

    if colunas_ausentes:
        raise ValueError(
            "O conjunto de dados não possui as colunas esperadas: "
            f"{sorted(colunas_ausentes)}"
        )

    return dados_carregados


def formatar_faixa_idade(intervalo: pd.Interval) -> str:
    """Transforma um intervalo do pandas em um rótulo legível."""
    inicio = max(0, round(float(intervalo.left)))
    fim = round(float(intervalo.right))

    return f"{inicio}–{fim} anos"


try:
    dados = carregar_dados()
except (OSError, ValueError) as erro:
    st.error(
        "Não foi possível carregar os dados do dashboard. "
        f"Detalhes: {erro}"
    )
    st.stop()


st.title("Dashboard do Titanic")

st.write(
    "Explore os dados dos passageiros e observe como a taxa de "
    "sobrevivência muda de acordo com os filtros selecionados."
)


st.sidebar.header("Filtros")

sexos_disponiveis = sorted(
    dados["Sex"].dropna().unique().tolist()
)

classes_disponiveis = sorted(
    dados["Pclass"].dropna().unique().tolist()
)

embarques_disponiveis = sorted(
    dados["Embarked"].dropna().unique().tolist()
)

sexos_selecionados = st.sidebar.multiselect(
    "Sexo",
    options=sexos_disponiveis,
    default=sexos_disponiveis,
    format_func=lambda valor: ROTULOS_SEXO.get(valor, valor),
)

classes_selecionadas = st.sidebar.multiselect(
    "Classe da passagem",
    options=classes_disponiveis,
    default=classes_disponiveis,
    format_func=lambda valor: ROTULOS_CLASSE.get(valor, valor),
)

embarques_selecionados = st.sidebar.multiselect(
    "Porto de embarque",
    options=embarques_disponiveis,
    default=embarques_disponiveis,
    format_func=lambda valor: ROTULOS_EMBARQUE.get(valor, valor),
)

aplicar_filtro_idade = st.sidebar.checkbox(
    "Aplicar filtro de idade"
)

idade_minima_dados = int(dados["Age"].dropna().min())
idade_maxima_dados = int(dados["Age"].dropna().max())

intervalo_idade = (
    idade_minima_dados,
    idade_maxima_dados,
)

if aplicar_filtro_idade:
    intervalo_idade = st.sidebar.slider(
        "Intervalo de idade",
        min_value=idade_minima_dados,
        max_value=idade_maxima_dados,
        value=intervalo_idade,
    )

variavel_analise = st.sidebar.selectbox(
    "Variável do gráfico principal",
    options=["Sex", "Pclass", "Embarked"],
    format_func=lambda valor: ROTULOS_VARIAVEIS[valor],
)

quantidade_faixas_idade = st.sidebar.slider(
    "Quantidade de faixas de idade",
    min_value=4,
    max_value=12,
    value=8,
    step=1,
)


dados_filtrados = dados[
    dados["Sex"].isin(sexos_selecionados)
    & dados["Pclass"].isin(classes_selecionadas)
    & dados["Embarked"].isin(embarques_selecionados)
].copy()

if aplicar_filtro_idade:
    idade_inicial, idade_final = intervalo_idade

    dados_filtrados = dados_filtrados[
        dados_filtrados["Age"].between(
            idade_inicial,
            idade_final,
        )
    ].copy()

if dados_filtrados.empty:
    st.warning(
        "Nenhum passageiro foi encontrado com os "
        "filtros selecionados."
    )
    st.stop()


total_passageiros = len(dados_filtrados)
total_sobreviventes = int(
    dados_filtrados["Survived"].sum()
)
total_nao_sobreviventes = (
    total_passageiros
    - total_sobreviventes
)
taxa_sobrevivencia = dados_filtrados["Survived"].mean()

(
    coluna_total,
    coluna_sobreviventes,
    coluna_nao_sobreviventes,
    coluna_taxa,
) = st.columns(4)

coluna_total.metric(
    "Passageiros",
    total_passageiros,
)

coluna_sobreviventes.metric(
    "Sobreviventes",
    total_sobreviventes,
)

coluna_nao_sobreviventes.metric(
    "Não sobreviventes",
    total_nao_sobreviventes,
)

coluna_taxa.metric(
    "Taxa de sobrevivência",
    f"{taxa_sobrevivencia:.2%}",
)


st.divider()

st.subheader(
    f"Análise por {ROTULOS_VARIAVEIS[variavel_analise]}"
)

dados_grafico = (
    dados_filtrados
    .groupby(
        variavel_analise,
        dropna=False,
    )
    .agg(
        Passageiros=("PassengerId", "count"),
        Sobreviventes=("Survived", "sum"),
        Taxa_sobrevivencia=("Survived", "mean"),
    )
    .reset_index()
)

dados_grafico["Taxa de sobrevivência (%)"] = (
    dados_grafico["Taxa_sobrevivencia"]
    * 100
)

if variavel_analise == "Sex":
    dados_grafico["Categoria"] = (
        dados_grafico["Sex"].map(ROTULOS_SEXO)
    )

elif variavel_analise == "Pclass":
    dados_grafico["Categoria"] = (
        dados_grafico["Pclass"].map(ROTULOS_CLASSE)
    )

else:
    dados_grafico["Categoria"] = (
        dados_grafico["Embarked"].map(ROTULOS_EMBARQUE)
    )

coluna_quantidade, coluna_sobrevivencia = st.columns(2)

with coluna_quantidade:
    st.markdown("#### Quantidade de passageiros")

    st.bar_chart(
        dados_grafico,
        x="Categoria",
        y="Passageiros",
        x_label=ROTULOS_VARIAVEIS[variavel_analise],
        y_label="Quantidade",
    )

with coluna_sobrevivencia:
    st.markdown("#### Taxa de sobrevivência")

    st.bar_chart(
        dados_grafico,
        x="Categoria",
        y="Taxa de sobrevivência (%)",
        x_label=ROTULOS_VARIAVEIS[variavel_analise],
        y_label="Sobrevivência (%)",
    )


st.divider()

st.subheader("Distribuição dos passageiros por idade")

dados_com_idade = dados_filtrados.dropna(
    subset=["Age"]
).copy()

if dados_com_idade.empty:
    st.info(
        "Não existem idades disponíveis para os "
        "filtros selecionados."
    )

elif dados_com_idade["Age"].nunique() == 1:
    idade_unica = dados_com_idade["Age"].iloc[0]

    distribuicao_idade = pd.DataFrame(
        {
            "Faixa de idade": [f"{idade_unica:g} anos"],
            "Passageiros": [len(dados_com_idade)],
            "Taxa de sobrevivência (%)": [
                dados_com_idade["Survived"].mean() * 100
            ],
        }
    )

else:
    dados_com_idade["Intervalo de idade"] = pd.cut(
        dados_com_idade["Age"],
        bins=quantidade_faixas_idade,
        include_lowest=True,
        precision=0,
        duplicates="drop",
    )

    distribuicao_idade = (
        dados_com_idade
        .groupby(
            "Intervalo de idade",
            observed=True,
            sort=True,
        )
        .agg(
            Passageiros=("PassengerId", "count"),
            Taxa_sobrevivencia=("Survived", "mean"),
        )
        .reset_index()
    )

    distribuicao_idade["Faixa de idade"] = (
        distribuicao_idade["Intervalo de idade"]
        .map(formatar_faixa_idade)
        .astype(str)
    )

    distribuicao_idade[
        "Taxa de sobrevivência (%)"
    ] = (
        distribuicao_idade["Taxa_sobrevivencia"]
        * 100
    )

coluna_idade, coluna_taxa_idade = st.columns(2)

with coluna_idade:
    st.markdown(
        "#### Passageiros por faixa de idade"
    )

    st.bar_chart(
        distribuicao_idade,
        x="Faixa de idade",
        y="Passageiros",
        x_label="Faixa de idade",
        y_label="Quantidade",
    )

with coluna_taxa_idade:
    st.markdown(
        "#### Sobrevivência por faixa de idade"
    )

    st.bar_chart(
        distribuicao_idade,
        x="Faixa de idade",
        y="Taxa de sobrevivência (%)",
        x_label="Faixa de idade",
        y_label="Sobrevivência (%)",
    )

st.caption(
    "As faixas são ordenadas numericamente. "
    "Quando uma barra de sobrevivência não aparece, "
    "a taxa daquele grupo é 0%."
)


st.divider()

with st.expander("Visualizar dados filtrados"):
    colunas_exibidas = [
        "PassengerId",
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]

    st.dataframe(
        dados_filtrados[colunas_exibidas],
        width="stretch",
        hide_index=True,
    )

st.caption(
    "Os gráficos e indicadores são atualizados automaticamente. "
    "Quando data/train.csv não está disponível, o dashboard usa "
    "uma cópia pública do conjunto Titanic hospedada no GitHub."
)
