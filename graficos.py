"""Arquivo de construção de gráficos para distribuições de probabilidade.
Construído para o módulo de Estatística do curso de Ciência de Dados ministrado
por Francisco Bustamante.

Pode ser modificado e distribuído livremente, desde que mantida a referência ao autor
a partir deste comentário de cabeçalho.

Autor: Francisco Bustamante
https://www.linkedin.com/in/flsbustamante
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, chi2, f, norm, t


def plot_pmf_discretas(x, y):

    plt.scatter(x, y, color="red")
    plt.vlines(x, 0, y, linewidth=1, linestyle="dotted")
    plt.xlim(x.min() - 0.5, x.max() + 0.5)
    plt.xticks(x)

    plt.show()


def plot_cdf_discretas(x, y):

    x_escala = np.arange(x.min(), x.max() + 2)

    plt.scatter(x, y, marker="o", color="red", zorder=2)

    plt.hlines(y=y, xmin=x_escala[1:], xmax=x_escala[:-1], zorder=1)
    plt.vlines(
        x=x_escala[1:-1],
        ymin=y[:-1],
        ymax=y[1:],
        color="red",
        linestyle="dashed",
        zorder=1,
        lw=0.5,
    )
    plt.scatter(
        x_escala[1:-1], y[:-1], color="white", marker="o", edgecolor="red", zorder=2
    )
    plt.xticks(x)
    plt.xlim(x.min() - 0.05, x.max() + 0.05)

    plt.show()


def plot_binomial(
    n_repeticoes=10, probabilidade_sucesso=0.5, n_experimentos=100_000, seed=42
):

    moeda = binom.rvs(
        n_repeticoes,
        probabilidade_sucesso,
        size=n_experimentos,
        random_state=seed,
    )
    valores, contagem = np.unique(moeda, return_counts=True)

    x = np.arange(0, n_repeticoes + 1)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # gráfico 1
    axs[0].vlines(valores, 0, contagem, lw=0.5, linestyles="dotted")
    axs[0].scatter(valores, contagem, marker="o", color="red")
    axs[0].set_xticks(x)
    axs[0].set_title("Contagem de sucessos")
    axs[0].set_xlim(x.min() - 0.05, x.max() + 0.05)

    # gráfico 2
    y = binom.pmf(x, n_repeticoes, probabilidade_sucesso)

    axs[1].vlines(x, 0, y, lw=0.5, linestyle="dotted")
    axs[1].scatter(x, y, marker="o", color="red")
    axs[1].set_xticks(x)
    axs[1].set_title("PMF da distribuição binomial")
    axs[1].set_xlim(x.min() - 0.05, x.max() + 0.05)

    # gráfico 3
    y = binom.cdf(x, n_repeticoes, probabilidade_sucesso)
    x_escala = np.arange(0, n_repeticoes + 2)

    axs[2].scatter(x, y, marker="o", color="red", zorder=2)
    axs[2].set_xticks(x)
    axs[2].set_title("CDF da distribuição binomial")
    axs[2].hlines(y=y, xmin=x_escala[1:], xmax=x_escala[:-1], zorder=1)
    axs[2].vlines(
        x=x_escala[1:-1],
        ymin=y[:-1],
        ymax=y[1:],
        color="red",
        linestyle="dashed",
        zorder=1,
        lw=0.5,
    )
    axs[2].scatter(
        x_escala[1:-1], y[:-1], color="white", marker="o", edgecolor="red", zorder=2
    )
    axs[2].set_xlim(x.min() - 0.05, x.max() + 0.05)

    fig.suptitle(
        f"Distribuição binomial com {n_repeticoes} repetições e probabilidade de {probabilidade_sucesso} de sucesso"
    )

    plt.show()


def plot_chi2(graus_de_liberdade, alfa):

    x = np.linspace(
        chi2.ppf(0.001, graus_de_liberdade), chi2.ppf(0.999, graus_de_liberdade), 1_000
    )
    y = chi2.pdf(x, graus_de_liberdade)

    valor_critico = chi2.ppf(1 - alfa, graus_de_liberdade)
    valor_acumulado = chi2.cdf(valor_critico, graus_de_liberdade)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(x, y)
    ax1.set_ylabel("pdf")
    ax1.fill_between(x, y, where=x > valor_critico, color="red", alpha=0.5)
    ax1.vlines(
        valor_critico,
        0,
        ymax=chi2.pdf(valor_critico, graus_de_liberdade),
        color="red",
        linestyle="--",
        label=f"Valor crítico: {valor_critico:.2f}",
    )

    ax2.plot(x, chi2.cdf(x, graus_de_liberdade))
    ax2.set_ylabel("cdf")
    ax2.fill_between(
        x,
        chi2.cdf(x, graus_de_liberdade),
        where=x < valor_critico,
        color="red",
        alpha=0.5,
    )
    ax2.vlines(
        valor_critico,
        0,
        ymax=valor_acumulado,
        color="red",
        linestyle="--",
        label=f"Valor crítico: {valor_critico:.2f}, cdf: {valor_acumulado:.2f}",
    )

    ax1.annotate(
        r"$\alpha$",
        xy=(valor_critico + 2, 0.01),
        xytext=(valor_critico + 5, 0.05),
        horizontalalignment="center",
        fontweight="bold",
        fontsize=14,
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    ax1.legend()
    ax2.legend()

    fig.suptitle(
        f"Distribuição Qui-quadrado para {graus_de_liberdade} graus de liberdade"
    )

    plt.show()


def plot_normal(mu, sigma, alfa, lado="ambos"):
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1_000)
    y = norm.pdf(x, mu, sigma)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(x, y)
    ax1.set_ylabel("pdf")

    ax2.plot(x, norm.cdf(x, mu, sigma))
    ax2.set_ylabel("cdf")

    if lado == "direita":
        valor_critico = norm.ppf(1 - alfa, mu, sigma)
        valor_acumulado = norm.cdf(valor_critico, mu, sigma)

        ax1.fill_between(x, y, where=x > valor_critico, color="red", alpha=0.5)
        ax1.vlines(
            valor_critico,
            0,
            ymax=norm.pdf(valor_critico, mu, sigma),
            color="red",
            linestyle="--",
            label=f"Valor crítico: {valor_critico:.2f}",
        )

        ax2.fill_between(
            x,
            0,
            norm.cdf(x, mu, sigma),
            where=x <= valor_critico,
            color="red",
            alpha=0.5,
        )
        ax2.vlines(
            valor_critico,
            0,
            ymax=1,
            color="red",
            linestyle="--",
            label=f"Valor crítico: {valor_critico:.2f}, cdf: {valor_acumulado:.2f}",
        )

        ax1.annotate(
            r"$\alpha$",
            xy=(valor_critico + sigma / 2, norm.pdf(valor_critico, mu, sigma) / 2),
            xytext=(valor_critico + 2 * sigma, norm.pdf(valor_critico, mu, sigma) / 2),
            horizontalalignment="center",
            fontweight="bold",
            fontsize=14,
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

    elif lado == "esquerda":
        valor_critico = norm.ppf(alfa, mu, sigma)
        valor_acumulado = norm.cdf(valor_critico, mu, sigma)

        ax1.fill_between(x, y, where=x < valor_critico, color="red", alpha=0.5)
        ax1.vlines(
            valor_critico,
            0,
            ymax=norm.pdf(valor_critico, mu, sigma),
            color="red",
            linestyle="--",
            label=f"Valor crítico: {valor_critico:.2f}",
        )

        ax2.fill_between(
            x,
            0,
            norm.cdf(x, mu, sigma),
            where=x <= valor_critico,
            color="red",
            alpha=0.5,
        )
        ax2.vlines(
            valor_critico,
            0,
            ymax=valor_acumulado,
            color="red",
            linestyle="--",
            label=f"Valor crítico: {valor_critico:.2f}, cdf: {valor_acumulado:.2f}",
        )

        ax1.annotate(
            r"$\alpha$",
            xy=(valor_critico - sigma / 2, norm.pdf(valor_critico, mu, sigma) / 2),
            xytext=(valor_critico - 2 * sigma, norm.pdf(valor_critico, mu, sigma) / 2),
            horizontalalignment="center",
            fontweight="bold",
            fontsize=14,
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

    elif lado == "ambos":
        valor_critico_direita = norm.ppf(1 - alfa / 2, mu, sigma)
        valor_critico_esquerda = norm.ppf(alfa / 2, mu, sigma)
        valor_acumulado_direita = norm.cdf(valor_critico_direita, mu, sigma)
        valor_acumulado_esquerda = norm.cdf(valor_critico_esquerda, mu, sigma)
        valor_acumulado = valor_acumulado_direita - valor_acumulado_esquerda

        ax1.fill_between(
            x,
            y,
            where=(x > valor_critico_direita) | (x < valor_critico_esquerda),
            color="red",
            alpha=0.5,
        )
        ax1.vlines(
            [valor_critico_direita, valor_critico_esquerda],
            0,
            ymax=[
                norm.pdf(valor_critico_direita, mu, sigma),
                norm.pdf(valor_critico_esquerda, mu, sigma),
            ],
            color="red",
            linestyle="--",
            label=f"Valor crítico: |{valor_critico_direita:.2f}|, cdf: {valor_acumulado:.2f}",
        )

        ax2.fill_between(
            x,
            0,
            norm.cdf(x, mu, sigma),
            where=(x >= valor_critico_esquerda) & (x <= valor_critico_direita),
            color="red",
            alpha=0.5,
        )
        ax2.vlines(
            [valor_critico_direita, valor_critico_esquerda],
            0,
            ymax=[valor_acumulado_direita, valor_acumulado_esquerda],
            color="red",
            linestyle="--",
            label=f"Valores crítico: |{valor_critico_direita:.2f}|, cdf: {valor_acumulado:.2f}",
        )

        ax1.annotate(
            r"$\alpha/2$",
            xy=(
                valor_critico_direita + sigma / 2,
                norm.pdf(valor_critico_direita, mu, sigma) / 2,
            ),
            xytext=(
                valor_critico_direita + 2 * sigma,
                norm.pdf(valor_critico_direita, mu, sigma) / 2,
            ),
            horizontalalignment="center",
            fontweight="bold",
            fontsize=14,
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

        ax1.annotate(
            r"$\alpha/2$",
            xy=(
                valor_critico_esquerda - sigma / 2,
                norm.pdf(valor_critico_esquerda, mu, sigma) / 2,
            ),
            xytext=(
                valor_critico_esquerda - 2 * sigma,
                norm.pdf(valor_critico_esquerda, mu, sigma) / 2,
            ),
            horizontalalignment="center",
            fontweight="bold",
            fontsize=14,
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

    ax1.legend()
    ax2.legend()

    fig.suptitle(f"Distribuição Normal para média {mu} e desvio padrão {sigma}")

    plt.show()


def plot_t_student(graus_de_liberdade, alfa, lado="ambos"):
    x = np.linspace(-4, 4, 1_000)
    y = t.pdf(x, graus_de_liberdade)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(x, y)
    ax1.set_ylabel("pdf")

    ax2.plot(x, t.cdf(x, graus_de_liberdade))
    ax2.set_ylabel("cdf")

    if lado == "direita":
        valor_critico = t.ppf(1 - alfa, graus_de_liberdade)
        valor_acumulado = t.cdf(valor_critico, graus_de_liberdade)

        ax1.fill_between(x, y, where=x > valor_critico, color="red", alpha=0.5)
        ax1.vlines(
            valor_critico,
            0,
            ymax=t.pdf(valor_critico, graus_de_liberdade),
            color="red",
            linestyle="--",
            label=f"Valor crítico: {valor_critico:.2f}",
        )

        ax2.fill_between(
            x,
            0,
            t.cdf(x, graus_de_liberdade),
            where=x <= valor_critico,
            color="red",
            alpha=0.5,
        )
        ax2.vlines(
            valor_critico,
            0,
            ymax=1,
            color="red",
            linestyle="--",
            label=f"Valor crítico: {valor_critico:.2f}, cdf: {valor_acumulado:.2f}",
        )

        ax1.annotate(
            r"$\alpha$",
            xy=(valor_critico + 0.5, t.pdf(valor_critico, graus_de_liberdade) / 2),
            xytext=(valor_critico + 1, t.pdf(valor_critico, graus_de_liberdade) / 2),
            horizontalalignment="center",
            fontweight="bold",
            fontsize=14,
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

    elif lado == "esquerda":
        valor_critico = t.ppf(alfa, graus_de_liberdade)
        valor_acumulado = t.cdf(valor_critico, graus_de_liberdade)

        ax1.fill_between(x, y, where=x < valor_critico, color="red", alpha=0.5)
        ax1.vlines(
            valor_critico,
            0,
            ymax=t.pdf(valor_critico, graus_de_liberdade),
            color="red",
            linestyle="--",
            label=f"Valor crítico: {valor_critico:.2f}",
        )

        ax2.fill_between(
            x,
            0,
            t.cdf(x, graus_de_liberdade),
            where=x <= valor_critico,
            color="red",
            alpha=0.5,
        )
        ax2.vlines(
            valor_critico,
            0,
            ymax=valor_acumulado,
            color="red",
            linestyle="--",
            label=f"Valor crítico: {valor_critico:.2f}, cdf: {valor_acumulado:.2f}",
        )

        ax1.annotate(
            r"$\alpha$",
            xy=(valor_critico - 0.5, t.pdf(valor_critico, graus_de_liberdade) / 2),
            xytext=(valor_critico - 1, t.pdf(valor_critico, graus_de_liberdade) / 2),
            horizontalalignment="center",
            fontweight="bold",
            fontsize=14,
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

    elif lado == "ambos":
        valor_critico_direita = t.ppf(1 - alfa / 2, graus_de_liberdade)
        valor_critico_esquerda = t.ppf(alfa / 2, graus_de_liberdade)
        valor_acumulado_direita = t.cdf(valor_critico_direita, graus_de_liberdade)
        valor_acumulado_esquerda = t.cdf(valor_critico_esquerda, graus_de_liberdade)
        valor_acumulado = valor_acumulado_direita - valor_acumulado_esquerda

        ax1.fill_between(
            x,
            y,
            where=(x > valor_critico_direita) | (x < valor_critico_esquerda),
            color="red",
            alpha=0.5,
        )
        ax1.vlines(
            [valor_critico_direita, valor_critico_esquerda],
            0,
            ymax=[
                t.pdf(valor_critico_direita, graus_de_liberdade),
                t.pdf(valor_critico_esquerda, graus_de_liberdade),
            ],
            color="red",
            linestyle="--",
            label=f"Valor crítico: |{valor_critico_direita:.2f}|, cdf: {valor_acumulado:.2f}",
        )

        ax2.fill_between(
            x,
            0,
            t.cdf(x, graus_de_liberdade),
            where=(x >= valor_critico_esquerda) & (x <= valor_critico_direita),
            color="red",
            alpha=0.5,
        )
        ax2.vlines(
            [valor_critico_direita, valor_critico_esquerda],
            0,
            ymax=[valor_acumulado_direita, valor_acumulado_esquerda],
            color="red",
            linestyle="--",
            label=f"Valores crítico: |{valor_critico_direita:.2f}|, cdf: {valor_acumulado:.2f}",
        )

        ax1.annotate(
            r"$\alpha/2$",
            xy=(
                valor_critico_direita + 0.5,
                t.pdf(valor_critico_direita, graus_de_liberdade) / 2,
            ),
            xytext=(
                valor_critico_direita + 1,
                t.pdf(valor_critico_direita, graus_de_liberdade) / 2,
            ),
            horizontalalignment="center",
            fontweight="bold",
            fontsize=14,
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

        ax1.annotate(
            r"$\alpha/2$",
            xy=(
                valor_critico_esquerda - 0.5,
                t.pdf(valor_critico_esquerda, graus_de_liberdade) / 2,
            ),
            xytext=(
                valor_critico_esquerda - 1,
                t.pdf(valor_critico_esquerda, graus_de_liberdade) / 2,
            ),
            horizontalalignment="center",
            fontweight="bold",
            fontsize=14,
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

    ax1.legend()
    ax2.legend()

    fig.suptitle(
        f"Distribuição t de Student com {graus_de_liberdade} graus de liberdade"
    )

    plt.show()


def plot_f_snedecor(graus_de_liberdade, alfa):

    x = np.linspace(
        f.ppf(0.001, *graus_de_liberdade), f.ppf(0.99, *graus_de_liberdade), 1_000
    )
    y = f.pdf(x, *graus_de_liberdade)

    valor_critico = f.ppf(1 - alfa, *graus_de_liberdade)
    valor_acumulado = f.cdf(valor_critico, *graus_de_liberdade)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(x, y)
    ax1.set_ylabel("pdf")
    ax1.fill_between(x, y, where=x > valor_critico, color="red", alpha=0.5)
    ax1.vlines(
        valor_critico,
        0,
        ymax=f.pdf(valor_critico, *graus_de_liberdade),
        color="red",
        linestyle="--",
        label=f"Valor crítico: {valor_critico:.2f}",
    )

    ax2.plot(x, f.cdf(x, *graus_de_liberdade))
    ax2.set_ylabel("cdf")
    ax2.fill_between(
        x,
        f.cdf(x, *graus_de_liberdade),
        where=x < valor_critico,
        color="red",
        alpha=0.5,
    )
    ax2.vlines(
        valor_critico,
        0,
        ymax=valor_acumulado,
        color="red",
        linestyle="--",
        label=f"Valor crítico: {valor_critico:.2f}, cdf: {valor_acumulado:.2f}",
    )

    ax1.annotate(
        r"$\alpha$",
        xy=(valor_critico + 1, 0.01),
        xytext=(valor_critico + 2, 0.05),
        horizontalalignment="center",
        fontweight="bold",
        fontsize=14,
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    ax1.legend()
    ax2.legend()

    fig.suptitle(f"Distribuição F para {graus_de_liberdade} graus de liberdade")

    plt.show()
