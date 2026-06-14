"""
main_baseline.py — Integrante 1
Este script realiza a preparação dos dados e o treinamento do modelo Baseline (Decision Tree) para o TDE 3. Ele inclui as seguintes etapas:

  1. Carregamento e pré-processamento do Fashion-MNIST
  2. Treinamento do modelo Baseline (Decision Tree, 784 features)
  3. Avaliação no conjunto de teste
  4. Exportação dos resultados para os demais integrantes
"""

from data.loader import carregar_fashion_mnist, preprocessar
from models.baseline import treinar, avaliar
from results.exporter import salvar_txt, salvar_npy


# ── Configuração ──────────────────────────────────────────
DATA_DIR = "."

OUTPUT_TXT = "results/output/baseline_resultados.txt"
OUTPUT_NPY = "results/output/baseline_resultados.npy"


def main():
    print("=" * 50)
    print("Preparação e Baseline")
    print("=" * 50)

    # 1. Carregamento
    X_train, y_train, X_test, y_test = carregar_fashion_mnist(DATA_DIR)

    # 2. Pré-processamento
    X_train_norm, X_test_norm = preprocessar(X_train, X_test)

    # 3. Treinamento
    clf, tempo_treino = treinar(X_train_norm, y_train)

    # 4. Avaliação
    acuracia, relatorio = avaliar(clf, X_test_norm, y_test)

    # 5. Exportação
    dados = {
        "abordagem":             "Baseline",
        "acuracia":              acuracia,
        "num_features":          784,
        "porcentagem_features":  100.0,
        "tempo_treino_s":        tempo_treino,
        "tempo_busca_s":         0.0,
        "relatorio":             relatorio,
    }

    salvar_txt(dados, OUTPUT_TXT)
    salvar_npy(dados, OUTPUT_NPY)

    # 6. Output no terminal
    print("\n" + "=" * 50)
    print("RESUMO FINAL — BASELINE")
    print("=" * 50)
    print(f"  Acurácia         : {acuracia * 100:.2f}%")
    print(f"  Features usadas  : 784 / 784 (100.00%)")
    print(f"  Tempo treino     : {tempo_treino:.4f} s")
    print(f"  Tempo busca      : N/A")
    print("=" * 50)

    print("\nRelatório por classe:")
    print(relatorio)


if __name__ == "__main__":
    main()
