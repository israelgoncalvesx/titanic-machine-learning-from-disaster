# Titanic Survival Prediction — Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Status](https://img.shields.io/badge/Status-em%20desenvolvimento-yellow)

Projeto de portfólio baseado no desafio **Titanic: Machine Learning from Disaster**, desenvolvido para praticar Análise de Dados, Machine Learning e organização de uma aplicação de Machine Learning Engineering.

## Aplicação publicada

A interface está disponível no Streamlit Community Cloud:

### [Acessar a aplicação Titanic ML](https://titanic-ml-israel.streamlit.app)

A aplicação possui duas páginas:

- **Previsão de sobrevivência:** recebe os dados de um passageiro e apresenta a classe prevista e as probabilidades estimadas pelo modelo;
- **Dashboard:** permite explorar os dados com filtros, indicadores e gráficos dinâmicos.

## Objetivo

O projeto busca responder à seguinte pergunta:

> **Com base nas características de um passageiro, é possível prever se ele sobreviveu ao Titanic?**

Além da previsão, o projeto foi utilizado para praticar:

- análise exploratória e preparação de dados;
- engenharia de atributos;
- tratamento de valores ausentes;
- codificação de variáveis categóricas;
- treinamento e avaliação de modelos;
- criação de um pipeline reproduzível;
- separação entre treinamento e inferência;
- persistência do modelo com Joblib;
- criação de uma interface com Streamlit;
- construção de um dashboard interativo;
- deploy no Streamlit Community Cloud.

## Fluxo do projeto

```text
Dados
  ↓
Análise exploratória
  ↓
Engenharia de atributos
  ↓
Pré-processamento
  ↓
Treinamento
  ↓
Avaliação
  ↓
Pipeline salvo
  ↓
Função de previsão
  ↓
Aplicação Streamlit
  ↓
Deploy
```

## Fonte dos dados

Os dados foram obtidos no Kaggle:

[**Titanic: Machine Learning from Disaster**](https://www.kaggle.com/datasets/shuofxz/titanic-machine-learning-from-disaster)

O projeto utiliza:

| Arquivo | Finalidade |
|---|---|
| `train.csv` | Treinamento, avaliação e dashboard |
| `test.csv` | Geração de previsões para submissão no Kaggle |

Para executar o treinamento localmente, coloque os arquivos em:

```text
data/train.csv
data/test.csv
```

Os CSVs não são versionados no repositório. Quando `data/train.csv` não está disponível, o dashboard usa uma cópia pública do conjunto de dados para continuar funcionando no deploy.

## Principais variáveis

| Variável | Descrição |
|---|---|
| `Survived` | Indica se o passageiro sobreviveu |
| `Pclass` | Classe da passagem |
| `Sex` | Sexo registrado |
| `Age` | Idade |
| `SibSp` | Quantidade de irmãos, irmãs ou cônjuges a bordo |
| `Parch` | Quantidade de pais ou filhos a bordo |
| `Fare` | Valor pago pela passagem |
| `Embarked` | Porto de embarque |

## Engenharia de atributos

Foram criadas duas características adicionais:

| Atributo | Descrição |
|---|---|
| `FamilySize` | Quantidade total de pessoas da família, incluindo o passageiro |
| `IsAlone` | Indica se o passageiro viajava sozinho |

O tamanho da família é calculado por:

```python
FamilySize = SibSp + Parch + 1
```

Essas transformações ficam em `src/features.py` e são reutilizadas no treinamento e nas previsões.

## Pré-processamento

### Variáveis numéricas

- `Age`;
- `SibSp`;
- `Parch`;
- `Fare`;
- `FamilySize`;
- `IsAlone`.

Tratamentos aplicados:

- `SimpleImputer` com a mediana;
- `StandardScaler`.

### Variáveis categóricas

- `Pclass`;
- `Sex`;
- `Embarked`.

Tratamentos aplicados:

- `SimpleImputer` com a categoria mais frequente;
- `OneHotEncoder`.

O pré-processamento e o modelo são reunidos em um único objeto `Pipeline` do Scikit-learn.

## Modelo e resultados

O modelo inicial é uma **Regressão Logística**, treinada com 80% dos dados e validada com os 20% restantes. A divisão utiliza `stratify=y` para preservar a proporção das classes.

A acurácia obtida no conjunto de validação foi de **81,56%**, em 179 passageiros.

| Classe | Precision | Recall | F1-score | Registros |
|---|---:|---:|---:|---:|
| Não sobreviveu | 0,82 | 0,90 | 0,86 | 110 |
| Sobreviveu | 0,81 | 0,68 | 0,74 | 69 |

O modelo apresentou maior facilidade para identificar passageiros que não sobreviveram. O principal ponto de melhoria é o recall da classe dos sobreviventes.

### Matriz de confusão

![Matriz de confusão do modelo](images/matriz_de_confusao.png)

O pipeline treinado é armazenado em:

```text
models/titanic_pipeline.joblib
```

## Interface Streamlit

O arquivo principal da aplicação é:

```text
app/streamlit_app.py
```

A interface permite informar:

- classe da passagem;
- sexo;
- idade;
- quantidade de irmãos, irmãs ou cônjuges;
- quantidade de pais ou filhos;
- valor da passagem;
- porto de embarque.

Ao selecionar **Realizar previsão**, a aplicação:

1. valida os dados informados;
2. cria `FamilySize` e `IsAlone`;
3. aplica o pipeline treinado;
4. apresenta a classe prevista;
5. exibe as probabilidades de sobreviver e não sobreviver.

## Dashboard interativo

A página do dashboard está em:

```text
app/pages/1_Dashboard.py
```

O dashboard possui:

- filtros por sexo;
- filtros por classe da passagem;
- filtros por porto de embarque;
- filtro opcional por intervalo de idade;
- escolha da variável analisada no gráfico principal;
- indicadores de passageiros, sobreviventes e taxa de sobrevivência;
- gráficos de quantidade e taxa de sobrevivência;
- distribuição por faixas de idade;
- tabela com os dados filtrados.

Os gráficos e indicadores são atualizados automaticamente quando os filtros são modificados.

## Como executar localmente

### 1. Clone o repositório

```bash
git clone https://github.com/israelgoncalvesx/titanic-machine-learning-from-disaster.git
cd titanic-machine-learning-from-disaster
```

### 2. Crie e ative o ambiente virtual

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
python -m pip install -r requirements.txt
```

### 4. Execute a aplicação Streamlit

Na raiz do projeto:

```bash
python -m streamlit run app/streamlit_app.py
```

O Streamlit exibirá um endereço local, normalmente:

```text
http://localhost:8501
```

Use a navegação lateral para alternar entre a página de previsão e o dashboard.

### 5. Execute o treinamento

Para treinar novamente o modelo, adicione os CSVs na pasta `data/` e execute:

```bash
python -m src.train
```

O comando prepara os dados, treina o pipeline, apresenta as métricas e salva o modelo em `models/titanic_pipeline.joblib`.

### 6. Execute uma previsão pelo terminal

```bash
python -m src.predict
```

Esse comando utiliza o passageiro de exemplo definido em `src/predict.py`.

## Observações importantes

- A previsão representa uma **estimativa estatística**, não uma certeza;
- as probabilidades exibidas se referem ao passageiro informado e não à acurácia geral do modelo;
- o modelo foi treinado com um conjunto de dados pequeno e histórico;
- a aplicação possui finalidade educacional e de portfólio;
- o modelo pode reproduzir padrões e limitações presentes nos dados de treinamento;
- o desempenho de 81,56% foi medido no conjunto de validação utilizado neste projeto;
- o resultado não deve ser interpretado como uma relação causal entre as variáveis e a sobrevivência.

## Limitações atuais

- O conjunto de dados é pequeno;
- a coluna `Cabin` possui muitos valores ausentes;
- o modelo utiliza apenas uma parte das informações disponíveis;
- o recall dos sobreviventes ainda pode ser melhorado;
- ainda não existe uma API externa para consumir o modelo;
- testes automatizados e Docker ainda não foram implementados;
- o desempenho no conjunto oficial de teste depende da avaliação do Kaggle.

## Estrutura atual

```text
titanic-machine-learning-from-disaster/
├── app/
│   ├── pages/
│   │   └── 1_Dashboard.py
│   └── streamlit_app.py
├── data/
│   ├── train.csv
│   └── test.csv
├── images/
│   └── matriz_de_confusao.png
├── models/
│   └── titanic_pipeline.joblib
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   └── 02_modelo_machine_learning.ipynb
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── predict.py
│   └── train.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## Tecnologias utilizadas

- Python;
- Pandas;
- NumPy;
- Matplotlib;
- Scikit-learn;
- Joblib;
- Streamlit;
- Jupyter Notebook;
- Git e GitHub;
- Streamlit Community Cloud.

Tecnologias planejadas:

- FastAPI;
- Pydantic;
- Pytest;
- Docker.

## Checklist do projeto

### Fundamentos e análise de dados

- [x] Criar o repositório e a estrutura inicial
- [x] Configurar ambiente virtual e dependências
- [x] Baixar os dados no Kaggle
- [x] Carregar `train.csv` e `test.csv`
- [x] Identificar tipos de dados e valores ausentes
- [x] Verificar registros duplicados
- [x] Produzir estatísticas descritivas
- [x] Analisar sobrevivência por sexo, classe, idade e embarque
- [x] Criar visualizações e interpretações

### Modelo no Jupyter Notebook

- [x] Definir `X` e `y`
- [x] Criar `FamilySize` e `IsAlone`
- [x] Separar dados de treino e validação
- [x] Tratar valores ausentes
- [x] Codificar variáveis categóricas
- [x] Criar um pipeline com Scikit-learn
- [x] Treinar uma Regressão Logística
- [x] Avaliar o modelo
- [x] Realizar validação cruzada
- [x] Gerar previsões para o conjunto de teste
- [x] Gerar `submission.csv`

### Organização para Machine Learning Engineering

- [x] Criar o pacote `src/`
- [x] Mover a engenharia de atributos para `src/features.py`
- [x] Criar `src/train.py`
- [x] Criar um pipeline reproduzível
- [x] Salvar o pipeline treinado com Joblib
- [x] Criar `src/predict.py`
- [x] Carregar o pipeline salvo sem novo treinamento
- [x] Gerar classe prevista e probabilidades
- [x] Transformar a previsão em uma função reutilizável
- [x] Validar os dados recebidos para previsão
- [x] Criar uma interface interativa com Streamlit
- [x] Criar um dashboard com filtros e gráficos dinâmicos
- [x] Preparar dependências e caminhos para o deploy
- [x] Publicar a aplicação no Streamlit Community Cloud
- [ ] Criar uma API com FastAPI
- [ ] Criar testes automatizados com Pytest
- [ ] Criar um `Dockerfile`
- [ ] Documentar a execução da API

### Melhorias opcionais

- [ ] Comparar Regressão Logística, Árvore de Decisão e Random Forest
- [ ] Ajustar hiperparâmetros com validação cruzada
- [ ] Extrair títulos da coluna `Name`
- [ ] Investigar informações de `Cabin` e `Ticket`
- [ ] Analisar a importância das características
- [ ] Registrar o resultado obtido no Kaggle

## Transparência sobre o uso de Inteligência Artificial

Este projeto foi desenvolvido com o auxílio de ferramentas de Inteligência Artificial, utilizadas para esclarecer dúvidas, revisar conceitos, organizar etapas e melhorar a documentação.

O código é executado, estudado e revisado pelo autor. As decisões, interpretações e conclusões são verificadas durante o processo de aprendizagem.

## Autor

**Israel Gabriel Gonçalves Almeida dos Santos**

- [LinkedIn](https://www.linkedin.com/in/israelgoncalvesx)
- [GitHub](https://github.com/israelgoncalvesx)
