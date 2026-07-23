# Titanic — Análise Exploratória e Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Pandas](https://img.shields.io/badge/Pandas-3.0-150458)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.9-F7931E)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![Status](https://img.shields.io/badge/Status-em%20desenvolvimento-yellow)

Projeto de portfólio em Ciência de Dados desenvolvido a partir do conjunto de dados **Titanic: Machine Learning from Disaster**.

O projeto percorre duas etapas do fluxo de trabalho de Ciência de Dados:

1. **Análise Exploratória de Dados (EDA):** investigação das características associadas à sobrevivência dos passageiros;
2. **Machine Learning:** construção de um pipeline de classificação para prever a sobrevivência dos passageiros presentes no conjunto de teste.

---

## Problema analisado

O objetivo principal é responder à seguinte pergunta:

> Quais características dos passageiros estavam associadas à sobrevivência no Titanic e como utilizar essas informações para construir um modelo de classificação?

A primeira parte do projeto busca compreender os dados, identificar padrões, verificar valores ausentes e produzir visualizações. A segunda transforma essas descobertas em variáveis utilizadas por um modelo de Machine Learning.

---

## Conjunto de dados

A base utilizada está disponível no Kaggle:

[Download do conjunto de dados](https://www.kaggle.com/datasets/shuofxz/titanic-machine-learning-from-disaster)

O projeto utiliza dois arquivos:

| Arquivo | Finalidade |
|---|---|
| `train.csv` | Contém os dados de 891 passageiros e a variável-alvo `Survived` |
| `test.csv` | Contém 418 passageiros sem a resposta de sobrevivência |

A variável-alvo é representada por:

- `0`: não sobreviveu;
- `1`: sobreviveu.

### Principais variáveis

| Variável | Descrição |
|---|---|
| `PassengerId` | Identificador do passageiro |
| `Survived` | Situação de sobrevivência |
| `Pclass` | Classe da passagem |
| `Name` | Nome do passageiro |
| `Sex` | Sexo registrado |
| `Age` | Idade |
| `SibSp` | Quantidade de irmãos, irmãs ou cônjuges a bordo |
| `Parch` | Quantidade de pais ou filhos a bordo |
| `Ticket` | Número do bilhete |
| `Fare` | Valor pago pela passagem |
| `Cabin` | Cabine registrada |
| `Embarked` | Porto de embarque |

---

## Estrutura do projeto

```text
titanic-machine-learning-from-disaster/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   └── 02_modelo_machine_learning.ipynb
├── .gitignore
├── enviroment.txt
├── README.md
└── requirements.txt
```

> Os arquivos CSV não são versionados no repositório. Eles devem ser baixados do Kaggle e colocados na pasta `data/`.

---

# Parte 1 — Análise Exploratória de Dados

Notebook: [`01_analise_exploratoria.ipynb`](notebooks/01_analise_exploratoria.ipynb)

A análise exploratória foi realizada para:

- conhecer as dimensões e os tipos das variáveis;
- identificar valores ausentes;
- verificar registros duplicados;
- calcular estatísticas descritivas;
- analisar a distribuição das idades e tarifas;
- comparar a sobrevivência por sexo, classe, idade e porto de embarque;
- comunicar os resultados por meio de tabelas, gráficos e interpretações.

## Qualidade dos dados

O conjunto de treino possui **891 registros e 12 variáveis**.

Não foram encontradas linhas completamente duplicadas.

As colunas com valores ausentes foram:

| Variável | Valores ausentes | Percentual aproximado |
|---|---:|---:|
| `Cabin` | 687 | 77,10% |
| `Age` | 177 | 19,87% |
| `Embarked` | 2 | 0,22% |

A coluna `Cabin` exige atenção especial devido à elevada ausência de dados. A idade pode ser tratada por imputação, enquanto `Embarked` possui apenas dois registros ausentes.

---

## Principais resultados da EDA

### Sobrevivência geral

Dos 891 passageiros:

| Situação | Quantidade | Percentual |
|---|---:|---:|
| Não sobreviveu | 549 | 61,62% |
| Sobreviveu | 342 | 38,38% |

A maioria dos passageiros presentes no conjunto de dados não sobreviveu.

### Sobrevivência por sexo

| Sexo | Taxa de sobrevivência |
|---|---:|
| Feminino | 74,20% |
| Masculino | 18,89% |

A diferença observada é expressiva. O sexo do passageiro apresentou forte associação com a sobrevivência.

### Sobrevivência por classe

| Classe | Taxa de sobrevivência |
|---|---:|
| 1ª classe | 62,96% |
| 2ª classe | 47,28% |
| 3ª classe | 24,24% |

Passageiros das classes superiores apresentaram taxas de sobrevivência maiores. A terceira classe teve a menor taxa.

### Distribuição das idades

A idade média foi de aproximadamente **29,70 anos**, enquanto a mediana foi de **28 anos**. A menor idade registrada foi de **0,42 ano** e a maior foi de **80 anos**.

A maior concentração de passageiros estava aproximadamente entre 18 e 35 anos.

### Sobrevivência por faixa etária

| Faixa etária | Taxa de sobrevivência |
|---|---:|
| Criança | 57,97% |
| Adolescente | 42,86% |
| Adulto jovem | 38,27% |
| Adulto | 40,00% |
| Idoso | 22,73% |

Crianças apresentaram a maior taxa de sobrevivência entre as faixas analisadas. Idosos tiveram a menor taxa.

Os 177 passageiros sem idade registrada não participaram dessa comparação.

### Tarifa e sobrevivência

Os passageiros sobreviventes apresentaram tarifas média e mediana superiores às dos não sobreviventes.

Esse resultado deve ser interpretado junto com a classe da passagem, pois passageiros de classes superiores geralmente pagavam tarifas maiores. A variável `Fare` também apresenta valores extremos, tornando a mediana uma medida importante para a interpretação.

### Sobrevivência por porto de embarque

| Porto | Código | Taxa de sobrevivência |
|---|---|---:|
| Cherbourg | `C` | 55,36% |
| Queenstown | `Q` | 38,96% |
| Southampton | `S` | 33,70% |

Cherbourg apresentou a maior taxa de sobrevivência. Entretanto, o porto de embarque também pode estar relacionado à classe e à tarifa dos passageiros.

---

## Visualizações produzidas

O notebook de análise exploratória contém os seguintes gráficos:

1. **Quantidade de passageiros por situação de sobrevivência** — compara sobreviventes e não sobreviventes;
2. **Taxa de sobrevivência por classe** — evidencia a diferença entre primeira, segunda e terceira classe;
3. **Histograma das idades** — apresenta a distribuição etária dos passageiros;
4. **Taxa de sobrevivência por faixa etária** — compara crianças, adolescentes, adultos jovens, adultos e idosos;
5. **Boxplot das tarifas por sobrevivência** — mostra medianas, dispersão e valores extremos;
6. **Taxa de sobrevivência por porto de embarque** — compara Cherbourg, Queenstown e Southampton.

Os gráficos e suas interpretações podem ser visualizados diretamente no notebook:

[`Abrir análise exploratória`](notebooks/01_analise_exploratoria.ipynb)

---

## Conclusões da análise exploratória

As variáveis que apresentaram associações mais relevantes com a sobrevivência foram:

- sexo;
- classe da passagem;
- faixa etária;
- tarifa;
- porto de embarque.

Mulheres, passageiros da primeira classe e crianças apresentaram taxas de sobrevivência maiores. Passageiros da terceira classe e idosos apresentaram taxas menores.

Essas relações são **associações observadas nos dados** e não provam causalidade. Algumas variáveis estão relacionadas entre si, como classe e tarifa.

---

# Parte 2 — Modelo de Machine Learning

Notebook: [`02_modelo_machine_learning.ipynb`](notebooks/02_modelo_machine_learning.ipynb)

A segunda parte utiliza o `train.csv` para treinar e validar um modelo de classificação. Depois da avaliação, o modelo é treinado com todos os registros e utilizado para gerar previsões para o `test.csv`.

## Engenharia de atributos

Foram criadas duas novas variáveis:

| Variável | Descrição |
|---|---|
| `FamilySize` | Total de familiares a bordo, incluindo o próprio passageiro |
| `IsAlone` | Indica se o passageiro viajava sozinho |

A variável `FamilySize` é calculada por:

```python
FamilySize = SibSp + Parch + 1
```

## Variáveis utilizadas pelo modelo

### Numéricas

- `Age`;
- `SibSp`;
- `Parch`;
- `Fare`;
- `FamilySize`;
- `IsAlone`.

### Categóricas

- `Pclass`;
- `Sex`;
- `Embarked`.

As colunas `PassengerId`, `Name`, `Ticket` e `Cabin` não foram utilizadas no primeiro modelo.

---

## Pré-processamento

O pré-processamento foi organizado com `Pipeline` e `ColumnTransformer`.

### Variáveis numéricas

1. preenchimento de valores ausentes com a mediana;
2. padronização com `StandardScaler`.

### Variáveis categóricas

1. preenchimento de valores ausentes com a categoria mais frequente;
2. transformação das categorias com `OneHotEncoder`.

O pipeline garante que as mesmas transformações sejam aplicadas aos conjuntos de treino, validação e teste, reduzindo inconsistências e risco de vazamento de dados.

---

## Modelo utilizado

O primeiro algoritmo escolhido foi a **Regressão Logística**.

Apesar do nome, a Regressão Logística é um algoritmo de classificação adequado para problemas binários, como prever:

- `0`: não sobreviveu;
- `1`: sobreviveu.

O conjunto de treino é dividido em:

- 80% para treinamento;
- 20% para validação.

A divisão utiliza `stratify=y`, mantendo proporções semelhantes de sobreviventes e não sobreviventes nos dois subconjuntos.

---

## Avaliação do modelo

O notebook calcula:

- acurácia;
- precisão;
- recall;
- F1-score;
- matriz de confusão;
- validação cruzada com cinco divisões.

A acurácia não é analisada isoladamente. O relatório de classificação e a matriz de confusão permitem avaliar se o modelo consegue identificar tanto sobreviventes quanto não sobreviventes.

> As métricas numéricas serão registradas no README depois que o notebook de Machine Learning for executado e validado no ambiente local.

---

## Arquivo de submissão

Depois do treinamento final, o notebook gera o arquivo:

```text
submission.csv
```

O arquivo possui exatamente duas colunas:

| Coluna | Descrição |
|---|---|
| `PassengerId` | Identificador do passageiro do conjunto de teste |
| `Survived` | Previsão produzida pelo modelo |

O notebook também valida automaticamente:

- quantidade de linhas;
- nomes e ordem das colunas;
- presença apenas dos valores `0` e `1`;
- correspondência dos identificadores com o `test.csv`.

---

# Tecnologias utilizadas

- Python;
- Pandas;
- NumPy;
- Matplotlib;
- Scikit-learn;
- Jupyter Notebook;
- Git e GitHub.

---

# Como executar o projeto

## 1. Clone o repositório

```bash
git clone https://github.com/israelgoncalvesx/titanic-machine-learning-from-disaster.git
cd titanic-machine-learning-from-disaster
```

## 2. Crie o ambiente virtual

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Linux ou macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Instale as dependências

```bash
python -m pip install -r requirements.txt
```

## 4. Adicione os dados

Baixe `train.csv` e `test.csv` no Kaggle e coloque os arquivos em:

```text
data/train.csv
data/test.csv
```

## 5. Abra os notebooks

```bash
jupyter notebook
```

Execute na seguinte ordem:

1. `notebooks/01_analise_exploratoria.ipynb`;
2. `notebooks/02_modelo_machine_learning.ipynb`.

Antes de finalizar, utilize:

```text
Restart Kernel → Run All
```

Isso verifica se todas as células funcionam quando executadas desde o início.

---

# Limitações

- O conjunto de dados é pequeno;
- `Cabin` possui aproximadamente 77% de valores ausentes;
- `Age` possui aproximadamente 20% de valores ausentes;
- as relações encontradas na EDA não comprovam causalidade;
- o modelo inicial não utiliza informações extraídas de `Name`, `Ticket` e `Cabin`;
- o desempenho no conjunto oficial de teste depende da avaliação realizada pelo Kaggle.

---

# Próximos passos

- executar e registrar as métricas finais da Regressão Logística;
- comparar o desempenho com Árvore de Decisão e Random Forest;
- ajustar hiperparâmetros com validação cruzada;
- extrair títulos da coluna `Name`, como `Mr`, `Mrs`, `Miss` e `Master`;
- extrair informações do convés a partir de `Cabin`;
- analisar grupos e famílias pelo número do `Ticket`;
- estudar importância das variáveis;
- publicar o resultado da submissão do Kaggle.

---

## Autor

**Israel Gabriel Gonçalves Almeida dos Santos**

- [LinkedIn](https://www.linkedin.com/in/israelgoncalvesx)
- [GitHub](https://github.com/israelgoncalvesx)
