# Titanic Survival Prediction — Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Pytest](https://img.shields.io/badge/Pytest-6%20passed-0A9EDC)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED)

[![Streamlit deployment check](https://github.com/israelgoncalvesx/titanic-machine-learning-from-disaster/actions/workflows/streamlit-check.yml/badge.svg)](https://github.com/israelgoncalvesx/titanic-machine-learning-from-disaster/actions/workflows/streamlit-check.yml)
[![Docker API validation](https://github.com/israelgoncalvesx/titanic-machine-learning-from-disaster/actions/workflows/docker-api-check.yml/badge.svg)](https://github.com/israelgoncalvesx/titanic-machine-learning-from-disaster/actions/workflows/docker-api-check.yml)

Projeto de portfólio baseado no desafio **Titanic: Machine Learning from Disaster**, desenvolvido para praticar Análise de Dados, Machine Learning e conceitos de Machine Learning Engineering.

O projeto evoluiu de notebooks exploratórios para uma solução com pipeline treinado, função de inferência reutilizável, interface Streamlit, API FastAPI, testes automatizados, contêiner Docker e integração contínua.

## Aplicações

### Interface Streamlit publicada

[Acessar a aplicação Titanic ML](https://titanic-ml-israel.streamlit.app)

A interface possui:

- formulário para previsão de sobrevivência;
- probabilidades produzidas pelo modelo;
- dashboard com filtros, indicadores e gráficos dinâmicos.

### API FastAPI

A API está pronta para publicação por meio de um Blueprint do Render:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/israelgoncalvesx/titanic-machine-learning-from-disaster)

O botão utiliza o arquivo `render.yaml` presente na raiz do repositório. A criação do serviço requer autorização do proprietário da conta Render para acessar o GitHub.

## Objetivo

O projeto busca responder à seguinte pergunta:

> **Com base nas características de um passageiro, é possível prever se ele sobreviveu ao Titanic?**

Também foram praticados:

- análise exploratória e preparação de dados;
- engenharia de atributos;
- tratamento de valores ausentes;
- codificação de variáveis categóricas;
- treinamento e avaliação de modelos;
- pipeline reproduzível com Scikit-learn;
- persistência do modelo com Joblib;
- separação entre treinamento e inferência;
- interface de usuário com Streamlit;
- API REST com FastAPI e Pydantic;
- testes automatizados com Pytest;
- conteinerização com Docker;
- integração contínua com GitHub Actions;
- preparação de deploy com Render Blueprint.

## Arquitetura

```text
Dados do Titanic
       ↓
Análise exploratória
       ↓
Engenharia de atributos
       ↓
Pré-processamento
       ↓
Regressão Logística
       ↓
Pipeline salvo com Joblib
       ↓
src.predict.prever_passageiro()
       ↓
 ┌───────────────┴───────────────┐
 ↓                               ↓
Streamlit                     FastAPI
Interface para pessoas        API para sistemas
                                 ↓
                              Pytest
                                 ↓
                              Docker
                                 ↓
                         GitHub Actions
                                 ↓
                              Render
```

A função `prever_passageiro()` concentra a inferência e é reutilizada pelo Streamlit e pela FastAPI. Isso evita duplicar o tratamento de dados e as regras de previsão.

## Fonte dos dados

Os dados foram obtidos no Kaggle:

[**Titanic: Machine Learning from Disaster**](https://www.kaggle.com/datasets/shuofxz/titanic-machine-learning-from-disaster)

| Arquivo | Finalidade |
|---|---|
| `train.csv` | Treinamento, avaliação e dashboard |
| `test.csv` | Geração de previsões para submissão no Kaggle |

Para treinar o modelo localmente, coloque os arquivos em:

```text
data/train.csv
data/test.csv
```

Os CSVs não são versionados. Quando `data/train.csv` não está disponível, o dashboard utiliza uma cópia pública do conjunto de dados.

## Variáveis utilizadas

| Variável | Descrição |
|---|---|
| `Pclass` | Classe da passagem |
| `Sex` | Sexo registrado |
| `Age` | Idade |
| `SibSp` | Irmãos, irmãs ou cônjuges a bordo |
| `Parch` | Pais ou filhos a bordo |
| `Fare` | Valor da passagem |
| `Embarked` | Porto de embarque |

### Engenharia de atributos

| Atributo | Descrição |
|---|---|
| `FamilySize` | Total de pessoas da família, incluindo o passageiro |
| `IsAlone` | Indica se o passageiro viajava sozinho |

```python
FamilySize = SibSp + Parch + 1
```

As transformações ficam em `src/features.py` e são utilizadas tanto no treinamento quanto na inferência.

## Pré-processamento

### Variáveis numéricas

- `Age`;
- `SibSp`;
- `Parch`;
- `Fare`;
- `FamilySize`;
- `IsAlone`.

Tratamentos:

- `SimpleImputer` com mediana;
- `StandardScaler`.

### Variáveis categóricas

- `Pclass`;
- `Sex`;
- `Embarked`.

Tratamentos:

- `SimpleImputer` com categoria mais frequente;
- `OneHotEncoder`.

O pré-processamento e a Regressão Logística são reunidos em um único `Pipeline`.

## Resultados do modelo

A Regressão Logística foi treinada com 80% dos dados e avaliada com os 20% restantes, usando `stratify=y`.

Acurácia no conjunto de validação: **81,56%**, em 179 passageiros.

| Classe | Precision | Recall | F1-score | Registros |
|---|---:|---:|---:|---:|
| Não sobreviveu | 0,82 | 0,90 | 0,86 | 110 |
| Sobreviveu | 0,81 | 0,68 | 0,74 | 69 |

O principal ponto de melhoria é o recall da classe dos sobreviventes.

### Matriz de confusão

![Matriz de confusão do modelo](images/matriz_de_confusao.png)

O pipeline treinado fica em:

```text
models/titanic_pipeline.joblib
```

## Interface Streamlit

Arquivo principal:

```text
app/streamlit_app.py
```

Dashboard:

```text
app/pages/1_Dashboard.py
```

A interface recebe os dados do passageiro, valida os valores, cria os atributos derivados, aplica o pipeline e exibe a classe e as probabilidades previstas.

## API FastAPI

Arquivo principal:

```text
api/main.py
```

### Rotas

| Método | Rota | Finalidade |
|---|---|---|
| `GET` | `/` | Informações básicas e status |
| `GET` | `/health` | Verificação de saúde |
| `POST` | `/predict` | Previsão de sobrevivência |
| `GET` | `/docs` | Documentação Swagger |
| `GET` | `/redoc` | Documentação ReDoc |

### Entrada de `/predict`

```json
{
  "Age": 29,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 80.0,
  "Pclass": 1,
  "Sex": "female",
  "Embarked": "C"
}
```

### Resposta de exemplo

```json
{
  "classe": 1,
  "resultado": "Sobreviveu",
  "probabilidade_nao_sobreviver": 0.0645,
  "probabilidade_sobreviver": 0.9355
}
```

A API rejeita automaticamente campos ausentes, campos extras, valores negativos e categorias fora dos valores aceitos.

## Testes automatizados

Arquivo:

```text
tests/test_api.py
```

Os seis testes verificam:

1. rota inicial;
2. rota de saúde;
3. previsão válida;
4. rejeição de idade negativa;
5. rejeição de classe inválida;
6. rejeição de campo adicional.

Resultado confirmado localmente com Python 3.13.7:

```text
6 passed
```

Execução:

```bash
python -m pytest -v
```

A API não precisa estar aberta para o `TestClient` executar os testes.

## Docker

Arquivos:

```text
Dockerfile
.dockerignore
```

O contêiner:

- utiliza `python:3.13-slim`;
- instala as dependências;
- copia a API, o código de inferência e o modelo;
- aceita a variável de ambiente `PORT`;
- inicia o Uvicorn em `0.0.0.0`;
- possui `HEALTHCHECK` baseado em `/health`.

### Construção local

```bash
docker build -t titanic-api .
```

### Execução local

```bash
docker run --rm -p 8000:8000 --name titanic-api titanic-api
```

Acesse:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## Integração contínua

### Validação do Streamlit

```text
.github/workflows/streamlit-check.yml
```

Verifica sintaxe, carregamento do modelo, fonte de dados do dashboard e inicialização do Streamlit.

### Validação da API Docker

```text
.github/workflows/docker-api-check.yml
```

A cada alteração relevante, o GitHub Actions:

1. constrói a imagem Docker;
2. inicia o contêiner;
3. aguarda a rota `/health` responder;
4. envia uma requisição para `/predict`;
5. valida as chaves e a soma das probabilidades;
6. apresenta os logs e remove o contêiner de teste.

O status atual pode ser consultado pelo badge **Docker API validation** no início deste README.

## Deploy da API no Render

A configuração está em:

```text
render.yaml
```

O Blueprint define:

- serviço web público;
- runtime Docker;
- plano gratuito;
- região `virginia`;
- branch `main`;
- health check em `/health`;
- deploy após aprovação das verificações de CI.

Para criar o serviço, clique no botão **Deploy to Render** no início do README, autorize o acesso ao GitHub, revise o Blueprint e confirme o deploy.

Depois da publicação, os endereços terão esta estrutura:

```text
API:          https://<nome-do-servico>.onrender.com
Documentação: https://<nome-do-servico>.onrender.com/docs
Saúde:        https://<nome-do-servico>.onrender.com/health
```

## Execução local sem Docker

### 1. Clone o repositório

```bash
git clone https://github.com/israelgoncalvesx/titanic-machine-learning-from-disaster.git
cd titanic-machine-learning-from-disaster
```

### 2. Crie o ambiente virtual

Windows:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Execute o Streamlit

```bash
python -m streamlit run app/streamlit_app.py
```

### 5. Execute a API

```bash
python -m uvicorn api.main:app --reload
```

### 6. Treine novamente o modelo

```bash
python -m src.train
```

### 7. Execute uma previsão pelo terminal

```bash
python -m src.predict
```

## Estrutura

```text
titanic-machine-learning-from-disaster/
├── .github/
│   └── workflows/
│       ├── docker-api-check.yml
│       └── streamlit-check.yml
├── api/
│   ├── __init__.py
│   └── main.py
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
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── render.yaml
└── requirements.txt
```

## Tecnologias

- Python 3.13;
- Pandas e NumPy;
- Matplotlib;
- Scikit-learn;
- Joblib;
- Streamlit;
- FastAPI, Pydantic e Uvicorn;
- Pytest e HTTPX;
- Docker;
- GitHub Actions;
- Render Blueprint;
- Jupyter Notebook;
- Git e GitHub.

## Observações e limitações

- A previsão representa uma estimativa estatística, não uma certeza;
- as probabilidades se referem ao passageiro enviado, não à acurácia geral;
- o modelo foi treinado com um conjunto pequeno e histórico;
- a aplicação tem finalidade educacional e de portfólio;
- o modelo pode reproduzir limitações dos dados de treinamento;
- o recall dos sobreviventes ainda pode ser melhorado;
- os testes atuais se concentram na API;
- o desempenho oficial no conjunto de teste depende da avaliação do Kaggle.

## Checklist

### Dados e modelo

- [x] Realizar análise exploratória
- [x] Criar atributos derivados
- [x] Criar pipeline de pré-processamento e modelo
- [x] Avaliar a Regressão Logística
- [x] Realizar validação cruzada
- [x] Gerar `submission.csv`
- [x] Salvar o pipeline com Joblib

### Aplicação e Machine Learning Engineering

- [x] Organizar o código em `src/`
- [x] Criar função reutilizável de inferência
- [x] Criar interface Streamlit
- [x] Criar dashboard interativo
- [x] Publicar o Streamlit Community Cloud
- [x] Criar API FastAPI
- [x] Validar entradas com Pydantic
- [x] Criar e aprovar seis testes com Pytest
- [x] Criar `Dockerfile` e `.dockerignore`
- [x] Configurar validação automatizada do Docker
- [x] Testar `/health` e `/predict` no workflow
- [x] Preparar o deploy no Render com `render.yaml`
- [ ] Autorizar a criação do serviço no Render
- [ ] Registrar no README a URL pública da API

### Melhorias opcionais

- [ ] Comparar outros algoritmos
- [ ] Ajustar hiperparâmetros
- [ ] Extrair títulos da coluna `Name`
- [ ] Investigar `Cabin` e `Ticket`
- [ ] Analisar importância das características
- [ ] Adicionar testes unitários para `src/features.py`
- [ ] Adicionar testes unitários para `src/predict.py`
- [ ] Registrar o resultado obtido no Kaggle

## Transparência sobre o uso de Inteligência Artificial

Este projeto foi desenvolvido com auxílio de ferramentas de Inteligência Artificial para esclarecer dúvidas, revisar conceitos, organizar etapas e melhorar a documentação.

O código é executado, estudado e revisado pelo autor. As decisões, interpretações e conclusões são verificadas durante o processo de aprendizagem.

## Autor

**Israel Gabriel Gonçalves Almeida dos Santos**

- [LinkedIn](https://www.linkedin.com/in/israelgoncalvesx)
- [GitHub](https://github.com/israelgoncalvesx)
