# Macacos Mordedores 🐒

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeoDiasTrindade/macacos_mordedores/blob/main/notebook/macacos_mordedores.ipynb)

Atividade prática desenvolvida para a apresentação **"Desmistificando a Inteligência Artificial"**, realizada no Ministério Interteen - Igreja Batista do Povo, Vila Mariana, para crianças de 11 a 13 anos, com objetivo de promover Letramento Digital em IA.

Depois de explicar de forma lúdica conceitos como aprendizado de máquina, overfitting/underfitting e árvores de decisão, a apresentação propõe um jogo: as crianças viram "tratadores de zoológico" e precisam descobrir, usando cartas com carinhas de macacos, quais mordem e quais não mordem, construindo suas próprias árvores de decisão a partir de dados de treino, e testando o modelo depois.

Este repositório contém também a versão em código Python mostrando como um computador resolveria esse problema de classificação na prática.

## Estrutura do repositório

- [`slides/`](slides/): a apresentação completa usada no dia, com a explicação dos conceitos e as instruções do jogo.
- [`notebook/`](notebook/): o notebook com a versão em código da atividade.
- [`data/`](data/): as imagens dos 40 macacos e a planilha com os rótulos (morde / não morde).
- [`src/macacos_mordedores/`](src/macacos_mordedores/utils.py): o código por trás do notebook. Fica escondido de propósito, pra manter o notebook focado no conceito e não na implementação.

<br><br><br>
---

Atividade "Macacos Mordedores" adaptada de IA Desplugada, de Annabel Lindner e Stefan
Seegerer (Friedrich-Alexander-Universität Erlangen-Nürnberg), tradução de Ricardo Lima Praciano de Sousa.
Disponível em https://www.aiunplugged.org — licenciado sob CC BY-NC 3.0.