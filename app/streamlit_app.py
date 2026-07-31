import streamlit as st

from src.predict import prever_passageiro


st.set_page_config(
    page_title="Previsão Titanic",
    page_icon="🚢",
    layout="centered",
)

st.title("Previsão de Sobrevivência no Titanic")

st.write(
    "Informe os dados de um passageiro para estimar "
    "a probabilidade de sobrevivência."
)

with st.form("formulario_passageiro"):
    classe = st.selectbox(
        "Classe da passagem",
        options=[1, 2, 3],
        format_func=lambda valor: f"{valor}ª classe",
    )

    sexo = st.selectbox(
        "Sexo",
        options=["female", "male"],
        format_func=lambda valor: {
            "female": "Feminino",
            "male": "Masculino",
        }[valor],
    )

    idade = st.number_input(
        "Idade",
        min_value=0,
        max_value=120,
        value=29,
        step=1,
    )

    irmaos_conjuges = st.number_input(
        "Quantidade de irmãos, irmãs ou cônjuges",
        min_value=0,
        value=0,
        step=1,
    )

    pais_filhos = st.number_input(
        "Quantidade de pais ou filhos",
        min_value=0,
        value=0,
        step=1,
    )

    tarifa = st.number_input(
        "Valor da passagem",
        min_value=0.0,
        value=80.0,
        step=1.0,
        format="%.2f",
    )

    porto_embarque = st.selectbox(
        "Porto de embarque",
        options=["S", "C", "Q"],
        format_func=lambda valor: {
            "S": "Southampton",
            "C": "Cherbourg",
            "Q": "Queenstown",
        }[valor],
    )

    botao_prever = st.form_submit_button(
        "Realizar previsão"
    )

if botao_prever:
    dados_passageiro = {
        "Age": idade,
        "SibSp": irmaos_conjuges,
        "Parch": pais_filhos,
        "Fare": tarifa,
        "Pclass": classe,
        "Sex": sexo,
        "Embarked": porto_embarque,
    }

    try:
        resultado = prever_passageiro(
            dados_passageiro
        )
    except (
        TypeError,
        ValueError,
        FileNotFoundError,
        OSError,
    ) as erro:
        st.error(str(erro))
        st.stop()

    st.subheader("Resultado da previsão")

    if resultado["classe"] == 1:
        st.success(
            "O passageiro provavelmente sobreviveria."
        )
    else:
        st.error(
            "O passageiro provavelmente não sobreviveria."
        )

    probabilidade_sobreviver = (
        resultado["probabilidade_sobreviver"]
    )

    probabilidade_nao_sobreviver = (
        resultado["probabilidade_nao_sobreviver"]
    )

    coluna_sobreviver, coluna_nao_sobreviver = st.columns(2)

    coluna_sobreviver.metric(
        "Probabilidade de sobreviver",
        f"{probabilidade_sobreviver:.2%}",
    )

    coluna_nao_sobreviver.metric(
        "Probabilidade de não sobreviver",
        f"{probabilidade_nao_sobreviver:.2%}",
    )

st.caption(
    "Esta aplicação tem finalidade educacional e demonstra "
    "o uso de um modelo de Machine Learning."
)
