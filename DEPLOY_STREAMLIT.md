# Deploy no Streamlit Community Cloud

Este projeto está preparado para publicação no Streamlit Community Cloud.

## Configuração do aplicativo

Use os seguintes valores ao criar o aplicativo:

| Campo | Valor |
|---|---|
| Repositório | `israelgoncalvesx/titanic-machine-learning-from-disaster` |
| Branch | `main` |
| Arquivo principal | `app/streamlit_app.py` |
| Versão do Python | `3.12` |

Não é necessário cadastrar secrets para a versão atual do projeto.

## Arquivos necessários

O deploy utiliza:

- `app/streamlit_app.py` como página principal;
- `app/pages/1_Dashboard.py` como dashboard interativo;
- `models/titanic_pipeline.joblib` como pipeline treinado;
- `requirements.txt` para instalar as dependências;
- `src/` para engenharia de atributos e inferência.

O dashboard procura primeiro por `data/train.csv`. Como os arquivos CSV locais não são versionados, ele usa automaticamente uma cópia pública do conjunto Titanic quando o arquivo local não está disponível.

## Publicação

1. Acesse o Streamlit Community Cloud.
2. Entre com a conta do GitHub.
3. Selecione **Create app**.
4. Escolha **Deploy a public app from GitHub**.
5. Informe o repositório, a branch e o arquivo principal da tabela acima.
6. Em **Advanced settings**, selecione Python 3.12.
7. Clique em **Deploy**.

## Atualizações futuras

Depois da publicação, novos commits enviados para a branch `main` são refletidos automaticamente na aplicação.

## Teste local antes de publicar

Na raiz do projeto, execute:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

Teste as duas páginas:

- previsão de sobrevivência;
- dashboard interativo.
