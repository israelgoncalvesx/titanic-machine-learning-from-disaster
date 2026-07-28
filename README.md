# Titanic Survival Prediction — Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E)
![Status](https://img.shields.io/badge/Status-em%20desenvolvimento-yellow)

> 🚧 **Projeto em desenvolvimento**
>
> Este projeto está sendo construído de forma gradual, com foco no aprendizado dos fundamentos de Machine Learning e na evolução de um notebook para uma aplicação organizada, reproduzível e preparada para receber uma API.

## Sobre o projeto

Este é um projeto de portfólio baseado no desafio **Titanic: Machine Learning from Disaster**.

O problema consiste em utilizar informações sobre os passageiros para construir um modelo capaz de prever se uma pessoa sobreviveu ou não ao naufrágio.

O projeto começou com uma Análise Exploratória de Dados em Jupyter Notebook e está sendo transformado, passo a passo, em um pequeno projeto de **Machine Learning Engineering**.

O fluxo planejado é:

```text
Dados
  ↓
Análise exploratória
  ↓
Preparação dos dados
  ↓
Treinamento do modelo
  ↓
Avaliação
  ↓
Modelo salvo
  ↓
API de previsão
  ↓
Testes e Docker
```

## Objetivo

O objetivo principal é desenvolver uma solução de classificação que responda à seguinte pergunta:

> **Com base nas características de um passageiro, é possível prever se ele sobreviveu ao Titanic?**

Além da previsão, o projeto busca praticar:

- análise e preparação de dados;
- separação entre características e variável-alvo;
- engenharia de atributos;
- treinamento e avaliação de modelos;
- organização de código Python fora do notebook;
- criação de um processo de treinamento reproduzível;
- disponibilização futura do modelo por meio de uma API;
- testes automatizados e conteinerização.

## Conceito central

Neste projeto:

```python
X = características utilizadas para fazer a previsão
y = resposta que o modelo precisa aprender
```

A variável-alvo é:

```python
y = dados["Survived"]
```

Onde:

- `0` representa um passageiro que não sobreviveu;
- `1` representa um passageiro que sobreviveu.

Algumas das características utilizadas em `X` são idade, sexo, classe da passagem, tarifa e porto de embarque.

## Conjunto de dados

A base pode ser encontrada no Kaggle:

[Download do conjunto de dados](https://www.kaggle.com/datasets/shuofxz/titanic-machine-learning-from-disaster)

O projeto utiliza:

| Arquivo | Finalidade |
|---|---|
| `train.csv` | Dados usados para treinamento e avaliação, incluindo a coluna `Survived` |
| `test.csv` | Dados sem a coluna `Survived`, utilizados para gerar previsões |

Os arquivos CSV não são versionados neste repositório. Depois de baixá-los, eles devem ser colocados na pasta `data/`.

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

Foram criadas duas novas características:

| Atributo | Descrição |
|---|---|
| `FamilySize` | Quantidade total de pessoas da família, incluindo o passageiro |
| `IsAlone` | Indica se o passageiro viajava sozinho |

O tamanho da família é calculado por:

```python
FamilySize = SibSp + Parch + 1
```

O número `1` representa o próprio passageiro.

A criação dessas características foi retirada do notebook e colocada em `src/features.py`, permitindo que a mesma transformação seja reutilizada no treinamento, nas previsões e, futuramente, na API.

## Etapa atual

O projeto está atualmente na transição entre:

```text
modelo desenvolvido no notebook
              ↓
código de treinamento em arquivos Python
```

O arquivo `src/train.py` já:

- carrega o `train.csv`;
- utiliza a função de engenharia de atributos;
- seleciona as características utilizadas pelo modelo;
- separa `X`, com as características, e `y`, com a resposta `Survived`;
- mostra as dimensões e as primeiras linhas dos dados.

O próximo passo será dividir os dados em treino e validação dentro do script.

## Checklist do projeto

### Fundamentos e análise de dados

- [x] Criar o repositório e a estrutura inicial
- [x] Configurar ambiente virtual e dependências
- [x] Carregar os arquivos `train.csv` e `test.csv`
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
- [x] Gerar o arquivo `submission.csv`

### Organização para Machine Learning Engineering

- [x] Criar o pacote `src/`
- [x] Mover a engenharia de atributos para `src/features.py`
- [x] Criar o arquivo `src/train.py`
- [x] Carregar os dados pelo script de treinamento
- [x] Separar as características `X` e a variável-alvo `y`
- [ ] Separar treino e validação no `src/train.py`
- [ ] Criar o pré-processamento no script
- [ ] Treinar o modelo pelo terminal
- [ ] Calcular e exibir métricas pelo script
- [ ] Salvar o pipeline treinado com `joblib`
- [ ] Criar um script para carregar o modelo e fazer previsões
- [ ] Validar os dados recebidos para previsão
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

## Estrutura atual

```text
titanic-machine-learning-from-disaster/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   └── 02_modelo_machine_learning.ipynb
├── src/
│   ├── __init__.py
│   ├── features.py
│   └── train.py
├── app/                       # API em uma etapa futura
├── models/                    # Pipeline treinado será salvo aqui
├── tests/                     # Testes automatizados em uma etapa futura
├── .gitignore
├── README.md
└── requirements.txt
```

## Análise exploratória

A análise exploratória está disponível em:

[`notebooks/01_analise_exploratoria.ipynb`](notebooks/01_analise_exploratoria.ipynb)

O conjunto de treino possui 891 passageiros e 12 colunas. Os principais valores ausentes estão em:

| Variável | Valores ausentes | Percentual aproximado |
|---|---:|---:|
| `Cabin` | 687 | 77,10% |
| `Age` | 177 | 19,87% |
| `Embarked` | 2 | 0,22% |

Entre os padrões observados na análise:

- mulheres apresentaram uma taxa de sobrevivência maior que os homens;
- passageiros da primeira classe apresentaram maior sobrevivência;
- passageiros da terceira classe apresentaram menor sobrevivência;
- crianças apresentaram uma taxa de sobrevivência relativamente maior;
- tarifa e classe da passagem apresentaram relações importantes entre si.

Essas relações representam associações observadas no conjunto de dados e não comprovam causalidade.

## Modelo inicial

O notebook de Machine Learning está disponível em:

[`notebooks/02_modelo_machine_learning.ipynb`](notebooks/02_modelo_machine_learning.ipynb)

O primeiro algoritmo utilizado foi a **Regressão Logística**, adequada para um problema de classificação binária.

O pré-processamento utiliza:

- `SimpleImputer` para valores ausentes;
- `StandardScaler` para características numéricas;
- `OneHotEncoder` para características categóricas;
- `ColumnTransformer` para organizar as transformações;
- `Pipeline` para unir o pré-processamento e o modelo.

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Git e GitHub

Tecnologias planejadas:

- Joblib
- FastAPI
- Pydantic
- Pytest
- Docker

## Como executar

### 1. Clone o repositório

```bash
git clone https://github.com/israelgoncalvesx/titanic-machine-learning-from-disaster.git
cd titanic-machine-learning-from-disaster
```

### 2. Crie e ative um ambiente virtual

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

### 4. Adicione os dados

Coloque os arquivos nos seguintes caminhos:

```text
data/train.csv
data/test.csv
```

### 5. Execute os notebooks

```bash
jupyter notebook
```

### 6. Execute o código Python atual

Na raiz do projeto:

```bash
python -m src.train
```

Atualmente, esse comando carrega a base, cria os novos atributos e separa `X` e `y`. O treinamento completo pelo terminal será implementado nas próximas etapas.

## Limitações atuais

- O conjunto de dados é pequeno;
- a coluna `Cabin` possui muitos valores ausentes;
- o primeiro modelo utiliza apenas uma parte das informações disponíveis;
- o treinamento completo ainda está concentrado no notebook;
- a API, os testes e o Docker ainda não foram implementados;
- o desempenho no conjunto oficial de teste depende da avaliação realizada pelo Kaggle.

## Transparência sobre o uso de Inteligência Artificial

Este projeto foi desenvolvido com o auxílio de ferramentas de Inteligência Artificial, utilizadas para esclarecer dúvidas, revisar conceitos, organizar etapas e melhorar a documentação.

O código é executado, estudado e revisado pelo autor. As decisões, interpretações e conclusões são verificadas durante o processo de aprendizagem.

## Autor

**Israel Gabriel Gonçalves Almeida dos Santos**

- [LinkedIn](https://www.linkedin.com/in/israelgoncalvesx)
- [GitHub](https://github.com/israelgoncalvesx)
