"""
results/exporter.py
Salva os resultados de qualquer abordagem (Baseline, Wrapper, GA)
em formatos reutilizáveis pelo Integrante 4.
"""

import os
import numpy as np


def salvar_txt(dados, caminho="resultados.txt"):
    """
    Salva um resumo legível dos resultados em arquivo .txt.

    Parâmetros
    ----------
    dados : dict com as chaves:
        - abordagem          (str)
        - acuracia           (float)
        - num_features       (int)
        - porcentagem_features (float)
        - tempo_treino_s     (float)
        - tempo_busca_s      (float)  — 0.0 se não houver etapa de busca
        - relatorio          (str)    — classification report (opcional)
    caminho : str
        Caminho do arquivo de saída.
    """
    os.makedirs(os.path.dirname(caminho) if os.path.dirname(caminho) else ".", exist_ok=True)

    with open(caminho, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f"RESULTADOS — {dados['abordagem'].upper()}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Modelo               : Decision Tree (random_state=1)\n")
        f.write(f"Features utilizadas  : {dados['num_features']} / 784\n")
        f.write(f"Porcentagem features : {dados['porcentagem_features']:.2f}%\n")
        f.write(f"Acurácia (teste)     : {dados['acuracia'] * 100:.2f}%\n")
        f.write(f"Tempo de treinamento : {dados['tempo_treino_s']:.4f} s\n")

        busca = dados.get("tempo_busca_s", 0.0)
        f.write(f"Tempo de busca       : {'N/A' if busca == 0.0 else f'{busca:.4f} s'}\n")

        if "relatorio" in dados:
            f.write("\nRelatório por classe:\n")
            f.write(dados["relatorio"])

        f.write("=" * 50 + "\n")

    print(f"[exporter] Resultados salvos em '{caminho}'")


def salvar_npy(dados, caminho="resultados.npy"):
    """
    Salva os valores numéricos em .npy para uso pelo Integrante 4.
    Carregue com: np.load(caminho, allow_pickle=True).item()
    """
    os.makedirs(os.path.dirname(caminho) if os.path.dirname(caminho) else ".", exist_ok=True)
    np.save(caminho, dados)
    print(f"[exporter] Dados numéricos salvos em '{caminho}'")
