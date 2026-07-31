# Titanic Survival Prediction — Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Pytest](https://img.shields.io/badge/Pytest-6%20passed-0A9EDC)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED)
![Status](https://img.shields.io/badge/Status-em%20desenvolvimento-yellow)

Projeto de portfólio baseado no desafio **Titanic: Machine Learning from Disaster**, desenvolvido para praticar Análise de Dados, Machine Learning e conceitos de Machine Learning Engineering.

O projeto evoluiu de notebooks exploratórios para uma solução organizada com pipeline treinado, função de inferência, interface Streamlit, API FastAPI, testes automatizados e preparação para execução em contêiner Docker.

## Aplicação publicada

A interface Streamlit está disponível no Streamlit Community Cloud:

### [Acessar a aplicação Titanic ML](https://titanic-ml-israel.streamlit.app)

A aplicação possui duas páginas:

- **Previsão de sobrevivência:** recebe os dados de um passageiro e apresenta a classe prevista e as probabilidades estimadas pelo modelo;
- **Dashboard:** permite explorar os dados com filtros, indicadores e gráficos dinâmicos.

> A interface Streamlit está publicada. A API FastAPI ainda é executada localmente e não possui um endereço público.

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
- persistência do modelo com Joblib;
- separação entre treinamento e inferência;
- criação de interface com Streamlit;
- disponibilização do modelo por meio de uma API REST;
- validação de dados com Pydantic;
- testes automatizados com Pytest;
- preparação da API para execução com Docker.

## Arquitetura atual

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
Interface para pessoas        API para outros sistemas
                                 ↓
                              Pytest
                       Testes automatizados
                                 ↓
                              Docker
                    Empacotamento da API
```

A função `prever_passageiro()` concentra a lógica de inferência e é reutilizada pelo Streamlit e pela FastAPI. Dessa forma, o projeto evita manter duas implementações diferentes para a mesma previsão.

## Fonte dos dados

Os dados foram obtidos no Kaggle:

[**Titanic: Machine Learning from Disaster**](https://www.kaggle.com/datasets/shuofxz/titanic-machine-learning-from-disaster)

| Arquivo | Finalidade |
|---|---|
| `train.csv` | Treinamento, avaliação e dashboard |
| `test.csv` | Geração de previsões para submissão no Kaggle |

Para executar o treinamento localmente, coloque os arquivos em:

```text
data/train.csv
data/test.csv
```

Os CSVs não são versionados no repositório. Quando `data/train.csv` não está disponível, o dashboard utiliza uma cópia pública do conjunto de dados para continuar funcionando no deploy.

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

Essas transformações ficam em `src/features.py` e são reutilizadas durante o treinamento e a inferência.

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

Ao selecionar **Realizar previsão**, a aplicação valida os dados, cria os atributos derivados, aplica o pipeline e exibe a classe prevista e as probabilidades.

## Dashboard interativo

A página do dashboard está em:

```text
app/pages/1_Dashboard.py
```

O dashboard possui filtros por sexo, classe, porto de embarque e idade, além de indicadores, gráficos dinâmicos e uma tabela com os dados filtrados.

## API FastAPI

A API está implementada em:

```text
api/main.py
```

Ela permite que outros programas utilizem o modelo por meio de requisições HTTP e respostas em JSON.

### Rotas disponíveis

| Método | Rota | Finalidade |
|---|---|---|
| `GET` | `/` | Informações básicas e status da API |
| `GET` | `/health` | Verificação de saúde da aplicação |
| `POST` | `/predict` | Realiza uma previsão de sobrevivência |
| `GET` | `/docs` | Documentação Swagger interativa |
| `GET` | `/redoc` | Documentação alternativa ReDoc |

### Dados esperados em `/predict`

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

### Exemplo de resposta

```json
{
  "classe": 1,
  "resultado": "Sobreviveu",
  "probabilidade_nao_sobreviver": 0.0645,
  "probabilidade_sobreviver": 0.9355
}
```

Os valores podem apresentar mais casas decimais. As probabilidades representam a estimativa do modelo para os dados enviados.

### Validações da API

A API rejeita automaticamente entradas como:

- idade menor que 0 ou maior que 120;
- classe diferente de 1, 2 ou 3;
- sexo diferente de `male` ou `female`;
- porto diferente de `S`, `C` ou `Q`;
- tarifa negativa;
- quantidades negativas em `SibSp` e `Parch`;
- campos obrigatórios ausentes;
- campos adicionais não previstos no esquema.

## Testes automatizados

Os testes estão em:

```text
tests/test_api.py
```

Foram implementados seis testes:

1. resposta da rota inicial;
2. resposta da rota de saúde;
3. previsão válida e probabilidades;
4. rejeição de idade negativa;
5. rejeição de classe inválida;
6. rejeição de campo adicional.

Resultado confirmado localmente com Python 3.13.7:

```text
6 passed
```

Os avisos apresentados pelas dependências durante a execução não causaram falha nos testes.

## Docker

O projeto possui:

```text
Dockerfile
.dockerignore
```

O `Dockerfile` utiliza Python 3.13 em uma imagem reduzida, instala as dependências, copia a API, o código de inferência e o modelo treinado, expõe a porta 8000 e inicia o Uvicorn.

O contêiner também possui uma verificação de saúde baseada na rota `/health`.

> O Dockerfile já foi criado. A construção da imagem e a execução do contêiner ainda precisam ser confirmadas localmente.

## Como executar localmente

### 1. Clone o repositório

```bash
git clone https://github.com/israelgoncalvesx/titanic-machine-learning-from-disaster.git
cd titanic-machine-learning-from-disaster
```

### 2. Crie o ambiente virtual

No Windows com Python 3.13:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Execute a aplicação Streamlit

```bash
python -m streamlit run app/streamlit_app.py
```

Endereço local padrão:

```text
http://localhost:8501
```

### 5. Execute a API FastAPI

```bash
python -m uvicorn api.main:app --reload
```

Endereços locais:

```text
API:          http://127.0.0.1:8000
Documentação: http://127.0.0.1:8000/docs
Saúde:        http://127.0.0.1:8000/health
```

### 6. Teste a API pelo PowerShell

```powershell
$passageiro = @{
    Age = 29
    SibSp = 0
    Parch = 0
    Fare = 80.0
    Pclass = 1
    Sex = "female"
    Embarked = "C"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $passageiro
```

### 7. Execute os testes

A API não precisa estar aberta para executar os testes:

```bash
python -m pytest -v
```

Resultado atual esperado:

```text
6 passed
```

### 8. Treine novamente o modelo

Adicione os CSVs na pasta `data/` e execute:

```bash
python -m src.train
```

### 9. Execute uma previsão pelo terminal

```bash
python -m src.predict
```

## Como executar com Docker

Com o Docker Desktop aberto, construa a imagem na raiz do projeto:

```bash
docker build -t titanic-api .
```

Execute o contêiner:

```bash
docker run --rm -p 8000:8000 --name titanic-api titanic-api
```

Depois acesse:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

Para interromper o contêiner em execução no terminal:

```text
Ctrl + C
```

Caso ele esteja sendo executado em segundo plano, utilize:

```bash
docker stop titanic-api
```

## Estrutura atual

```text
titanic-machine-learning-from-disaster/
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
└── requirements.txt
```

## Tecnologias utilizadas

- Python 3.13;
- Pandas;
- NumPy;
- Matplotlib;
- Scikit-learn;
- Joblib;
- Streamlit;
- FastAPI;
- Pydantic;
- Uvicorn;
- Pytest;
- HTTPX;
- Docker;
- Jupyter Notebook;
- Git e GitHub;
- Streamlit Community Cloud.

## Observações importantes

- A previsão representa uma **estimativa estatística**, não uma certeza;
- as probabilidades se referem ao passageiro informado e não à acurácia geral do modelo;
- o modelo foi treinado com um conjunto de dados pequeno e histórico;
- o projeto possui finalidade educacional e de portfólio;
- o modelo pode reproduzir padrões e limitações dos dados de treinamento;
- o desempenho de 81,56% foi medido no conjunto de validação deste projeto;
- o resultado não deve ser interpretado como uma relação causal entre as variáveis e a sobrevivência.

## Limitações atuais

- O conjunto de dados é pequeno;
- a coluna `Cabin` possui muitos valores ausentes;
- o modelo utiliza apenas uma parte das informações disponíveis;
- o recall dos sobreviventes ainda pode ser melhorado;
- a API ainda não está publicada em um servidor público;
- a imagem Docker ainda precisa ser validada localmente;
- os testes atuais estão concentrados nas rotas da API;
- o desempenho no conjunto oficial de teste depende da avaliação do Kaggle.

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
- [x] Criar uma API com FastAPI
- [x] Validar entradas da API com Pydantic
- [x] Criar testes automatizados com Pytest
- [x] Executar e aprovar os seis testes da API
- [x] Criar um `Dockerfile`
- [x] Criar um `.dockerignore`
- [x] Documentar a execução da API
- [x] Documentar a execução dos testes
- [x] Documentar a execução com Docker
- [ ] Construir e validar a imagem Docker localmente
- [ ] Publicar a API em um serviço de nuvem

### Melhorias opcionais

- [ ] Comparar Regressão Logística, Árvore de Decisão e Random Forest
- [ ] Ajustar hiperparâmetros com validação cruzada
- [ ] Extrair títulos da coluna `Name`
- [ ] Investigar informações de `Cabin` e `Ticket`
- [ ] Analisar a importância das características
- [ ] Registrar o resultado obtido no Kaggle
- [ ] Adicionar testes unitários para `src/features.py`
- [ ] Adicionar testes unitários para `src/predict.py`
- [ ] Configurar integração contínua para executar o Pytest

## Transparência sobre o uso de Inteligência Artificial

Este projeto foi desenvolvido com o auxílio de ferramentas de Inteligência Artificial, utilizadas para esclarecer dúvidas, revisar conceitos, organizar etapas e melhorar a documentação.

O código é executado, estudado e revisado pelo autor. As decisões, interpretações e conclusões são verificadas durante o processo de aprendizagem.

## Autor

**Israel Gabriel Gonçalves Almeida dos Santos**

- [LinkedIn](https://www.linkedin.com/in/israelgoncalvesx)
- [GitHub](https://github.com/israelgoncalvesx)
