# CONFIG
import re
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

TAMANHO = 4
PASTA_IMAGENS = '../data'

def carregar_como_numeros(caminho):
    imagem = Image.open(caminho).convert("L").resize((TAMANHO, TAMANHO))
    return np.array(imagem).flatten() / 255.0

def mostrar_macaco_como_a_arvore_ve(macaco='macaco_01'):
    caminho = f"{PASTA_IMAGENS}/{macaco}.png"
    img_colorida = Image.open(caminho).convert("RGB")
    img_cinza = np.array(Image.open(caminho).convert("L").resize((TAMANHO, TAMANHO)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Esquerda: a imagem original, colorida
    ax1.imshow(img_colorida)
    ax1.set_title(f"{macaco} (como nós enxergamos)")
    ax1.axis("off")

    # Direita: versão em cinza com os números em cada quadradinho
    ax2.imshow(img_cinza, cmap="gray")
    ax2.set_title("O que o computador enxerga (números!)")
    ax2.axis("off")
    for i in range(img_cinza.shape[0]):
        for j in range(img_cinza.shape[1]):
            valor = img_cinza[i, j]
            cor = "black" if valor > 128 else "white"
            ax2.text(j, i, str(valor), ha="center", va="center",
                     color=cor, fontsize=15)

    plt.tight_layout()
    plt.show()

def montar_conjunto(subset):
    X = np.array([carregar_como_numeros(c) for c in subset.arquivo])
    y = subset.rotulo_numero.to_numpy()
    numeros = subset.numero.tolist()
    return X, y, numeros

def gerar_arvore(modelo):
    nomes_das_perguntas = [f"Pixel (linha {i // TAMANHO + 1}, coluna {i % TAMANHO + 1})"
                           for i in range(TAMANHO * TAMANHO)]

    plt.figure(figsize=(16, 8))
    anotacoes = plot_tree(modelo, filled=True, rounded=True,
            class_names=["não morde", "morde"],
            feature_names=nomes_das_perguntas,
            fontsize=12, impurity=False, proportion=True,
            label='none')

    def converte_limiar(match):
        valor = float(match.group(1)) * 255
        return f"< {valor:.0f}"

    for texto in anotacoes:
        conteudo = texto.get_text()

        # troca True/False por sim/não (setas da raiz)
        conteudo = conteudo.replace("True", "sim").replace("False", "não")

        # converte o limiar de 0-1 para 0-255
        conteudo = re.sub(r"<= (\d+\.?\d*)", converte_limiar, conteudo)

        linhas = conteudo.split("\n")
        if "<=" in conteudo:
            linhas = [l for l in linhas if l not in ("morde", "não morde")]
        texto.set_text("\n".join(linhas))


def criando_modelo(X_validacao, y_validacao, MAX_PERGUNTAS):
    modelo = DecisionTreeClassifier(
        max_depth=MAX_PERGUNTAS,
        random_state=24,
        class_weight={0: 1, 1: 1}
    )
    modelo.fit(X_validacao, y_validacao)
    print("Árvore treinada com os macacos de validação")

    gerar_arvore(modelo)

    return modelo


def mostrar_pixels_da_arvore(modelo, macaco='macaco_01'):
    caminho = f"{PASTA_IMAGENS}/{macaco}.png"
    img_original = Image.open(caminho).convert("RGB")
    largura, altura = img_original.size
    bloco_w = largura / TAMANHO
    bloco_h = altura / TAMANHO

    valores = carregar_como_numeros(caminho)

    tree = modelo.tree_
    plt.figure(figsize=(4, 3))
    plt.imshow(img_original)

    for no in range(tree.node_count):
        p = tree.feature[no]
        if p == -2:
            continue
        limiar = tree.threshold[no]
        valor = valores[p]
        resposta_sim = valor <= limiar

        # conversão para escala de cinza 0-255 (só para exibição)
        valor_255 = valor * 255
        limiar_255 = limiar * 255

        linha, coluna = p // TAMANHO, p % TAMANHO
        x, y = coluna * bloco_w, linha * bloco_h
        cor = "lime" if resposta_sim else "red"

        plt.gca().add_patch(plt.Rectangle((x, y), bloco_w, bloco_h,
                            fill=False, edgecolor=cor, linewidth=3))
        plt.text(x + bloco_w / 2, y + bloco_h / 2,
                 f"{valor_255:.0f}\n(<{limiar_255:.0f}?)",
                 color=cor, ha="center", va="center",
                 fontsize=5, fontweight="bold",
                 bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))

    plt.axis("off")

def usando_o_modelo_com_macacos_de_teste(modelo, macacos, X_teste, y_teste, numeros_teste):
    previsoes = modelo.predict(X_teste)

    rotulo_txt = {0: "não morde", 1: "morde"}

    n = len(numeros_teste)
    colunas = 5
    linhas_grade = int(np.ceil(n / colunas))

    fig, eixos = plt.subplots(linhas_grade, colunas, figsize=(3 * colunas, 3.4 * linhas_grade))
    eixos = eixos.flatten()

    pontuacao = 0

    for i, (numero, real, previsto) in enumerate(zip(numeros_teste, y_teste, previsoes)):
        ax = eixos[i]
        caminho = macacos.loc[macacos.numero == numero, "arquivo"].values[0]
        imagem = Image.open(caminho)
        ax.imshow(imagem)
        ax.axis("off")

        if real == previsto:
            pontos = +1
            situacao = "ACERTOU (+1)\n"
            cor_moldura = "green"
        elif previsto == 1:
            pontos = -1
            situacao = "ERROU: ALARME FALSO\n"
            cor_moldura = "orange"
        else:
            pontos = -3
            situacao = "ERROU: MORDIDA!\n"
            cor_moldura = "red"

        pontuacao += pontos

        for lado in ax.spines.values():
            lado.set_visible(True)
            lado.set_color(cor_moldura)
            lado.set_linewidth(5)
        ax.set_xticks([]); ax.set_yticks([])

        ax.text(0.5, 1.1, situacao, transform=ax.transAxes, ha="center", va="top",
                fontsize=11, fontweight="bold", color=cor_moldura)
        
    for j in range(i + 1, len(eixos)):
        eixos[j].axis("off")
    plt.tight_layout()

    print(f"Pontuação final: {pontuacao} de {n} pontos possíveis.")
    plt.show()

    return pontuacao

def carregar_dados(test_size=0.5, random_state=24):
    macacos = pd.read_csv(f'{PASTA_IMAGENS}/macacos_rotulos.csv')

    treino, teste = train_test_split(
        macacos,
        test_size=0.5,
        shuffle=False,
    )

    X_treino, y_treino, num_treino = montar_conjunto(treino)
    X_teste, y_teste, num_teste = montar_conjunto(teste)

    return macacos, X_treino, y_treino, num_treino, X_teste, y_teste, num_teste