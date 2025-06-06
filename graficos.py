import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurar saída UTF-8
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Define estilo visual moderno
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Caminho de saída para salvar os gráficos
saida = r"C:\Users\gianlluca\Desktop\Faculdade\Matérias\10º Semestre\Inteligência Computacional\Exercícios\Trabalho RNA\trabalho-rna\graficos"
os.makedirs(saida, exist_ok=True)

# 1. Carregar os dados
df = pd.read_csv("train.csv")

def gerar_histograma_grlivarea(df, saida):
    """
    Gera o histograma da variável GrLivArea com destaque para outliers acima de 4000.
    """
    plt.figure()
    sns.histplot(data=df, x="GrLivArea", kde=True, bins=40, color="cornflowerblue")
    plt.title("Figura 1: Outliers com base em GrLivArea")
    plt.xlabel("GrLivArea (Área construída)")
    plt.ylabel("Frequência")
    plt.axvline(4000, color="red", linestyle="--", label="Limite superior (4000)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(saida, "histograma_grlivarea_outliers.png"), dpi=300)
    print("✅ Figura 1 salva: histograma_grlivarea_outliers.png")

def gerar_histograma_saleprice(df, saida):
    """
    Gera o histograma da variável SalePrice com marcação do percentil 99,5.
    """
    plt.figure()
    sns.histplot(data=df, x="SalePrice", kde=True, bins=40, color="mediumseagreen")
    plt.title("Figura 2: Outliers na variável SalePrice")
    plt.xlabel("SalePrice")
    plt.ylabel("Frequência")
    plt.axvline(df["SalePrice"].quantile(0.995), color="red", linestyle="--", label="Percentil 99,5")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(saida, "histograma_saleprice_outliers.png"), dpi=300)
    print("✅ Figura 2 salva: histograma_saleprice_outliers.png")

def gerar_grafico_normalizacao_saleprice(df, saida):
    """
    Gera um gráfico com dois histogramas lado a lado:
    - SalePrice original
    - SalePrice após transformação log1p + padronização
    """
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Preparar dados
    saleprice_original = df["SalePrice"]
    saleprice_log = np.log1p(saleprice_original)
    saleprice_scaled = StandardScaler().fit_transform(saleprice_log.values.reshape(-1, 1)).flatten()

    # Figura 3: Distribuição antes e depois da transformação
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Esquerda: original
    sns.histplot(saleprice_original, bins=40, kde=True, ax=axes[0], color="tomato")
    axes[0].set_title("Antes da Transformação")
    axes[0].set_xlabel("SalePrice")
    axes[0].set_ylabel("Frequência")

    # Direita: transformada
    sns.histplot(saleprice_scaled, bins=40, kde=True, ax=axes[1], color="steelblue")
    axes[1].set_title("Após log1p + StandardScaler")
    axes[1].set_xlabel("SalePrice (transformado)")
    axes[1].set_ylabel("Frequência")

    plt.suptitle("Figura 3: Distribuição de SalePrice antes e depois da transformação", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(saida, "normalizacao_saleprice.png"), dpi=300)
    print("✅ Figura 3 salva: normalizacao_saleprice.png")

def gerar_curva_loss_otimizadores(saida, caminho_antigo, caminho_atual):
    """
    Gera a Figura 4 a partir de logs reais (CSV):
    - caminho_antigo: CSV com colunas 'epoch' e 'val_loss' (SGD)
    - caminho_atual: CSV com colunas 'epoch' e 'val_loss' (Adam)
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Carregar os logs
    df_sgd = pd.read_csv(caminho_antigo)
    df_adam = pd.read_csv(caminho_atual)

    df_sgd = df_sgd[df_sgd["epoch"] <= 1000]
    df_adam = df_adam[df_adam["epoch"] <= 1000]

    # Plotar a curva de perda de validação
    plt.figure(figsize=(10, 6))
    plt.plot(df_sgd["epoch"], df_sgd["val_loss"], label="SGD (modelo antigo)", color="orange", linewidth=2)
    plt.plot(df_adam["epoch"], df_adam["val_loss"], label="Adam (modelo atual)", color="royalblue", linewidth=2)

    plt.title("Figura 4: Comparação da curva de perda com SGD e Adam")
    plt.xlabel("Épocas")
    plt.ylabel("Loss de Validação")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Salvar imagem
    output_path = os.path.join(saida, "loss_otimizadores.png")
    plt.savefig(output_path, dpi=300)
    print(f"✅ Figura 4 salva: {output_path}")

if __name__ == "__main__":

    #gerar_histograma_grlivarea(df, saida)
    #gerar_histograma_saleprice(df, saida)
    #gerar_grafico_normalizacao_saleprice(df, saida)

    saida = r"C:\Users\gianlluca\Desktop\Faculdade\Matérias\10º Semestre\Inteligência Computacional\Exercícios\Trabalho RNA\trabalho-rna\graficos"
    log_sgd = r"C:\Users\gianlluca\Desktop\Faculdade\Matérias\10º Semestre\Inteligência Computacional\Exercícios\Trabalho RNA\trabalho-rna\loss_dados\log_codigo_antigo.csv"
    log_adam = r"C:\Users\gianlluca\Desktop\Faculdade\Matérias\10º Semestre\Inteligência Computacional\Exercícios\Trabalho RNA\trabalho-rna\loss_dados\log_codigo_organizado.csv"
    
    gerar_curva_loss_otimizadores(saida, log_sgd, log_adam)