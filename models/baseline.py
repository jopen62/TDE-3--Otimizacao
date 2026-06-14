"""
Treina e avalia o modelo Baseline com 100% das features (784).
"""

import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from data.loader import CLASSES


def treinar(X_train, y_train):
    """
    Treina uma Decision Tree com todas as features disponíveis.

    Parâmetros conforme enunciado:
    - Algoritmo : DecisionTreeClassifier (scikit-learn)
    - random_state=1, demais parâmetros padrão

    Retorna: 
    clf : modelo treinado
    tempo_treino: float — tempo de treinamento em segundos
    """
    clf = DecisionTreeClassifier(random_state=1)

    print("[baseline] Treinando Decision Tree com 784 features...")
    inicio = time.time()
    clf.fit(X_train, y_train)
    tempo_treino = time.time() - inicio

    print(f"[baseline] Treinamento concluído em {tempo_treino:.4f} s")
    return clf, tempo_treino


def avaliar(clf, X_test, y_test):
    """
    Avalia o modelo no conjunto de teste.

    Retorna: 
    acuracia : float
    relatorio: str — classification report por classe
    """
    y_pred = clf.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred, target_names=CLASSES)

    print(f"[baseline] Acurácia no conjunto de teste: {acuracia * 100:.2f}%")
    return acuracia, relatorio
