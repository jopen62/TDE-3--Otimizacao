"""
Responsável por carregar e pré-processar o dataset Fashion-MNIST.
"""

import os
import struct
import numpy as np


# Arquivos do Fashion-MNIST (formato Kaggle/Zalando)
ARQUIVOS = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images":  "t10k-images-idx3-ubyte",
    "test_labels":  "t10k-labels-idx1-ubyte",
}

# Mapeamento das classes do Fashion-MNIST
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def _load_images(filepath):
    """
    Lê o arquivo binário de imagens no formato IDX.
    Retorna array numpy de shape (N, 784) com valores uint8.
    """
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Magic number inválido: {magic}"
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
    return images


def _load_labels(filepath):
    """
    Lê o arquivo binário de labels no formato IDX.
    Retorna array numpy de shape (N,) com valores [0-9].
    """
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Magic number inválido: {magic}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def carregar_fashion_mnist(data_dir="."):
    """
    Carrega os quatro arquivos do Fashion-MNIST a partir de data_dir.

    Parâmetros:
    data_dir : str
    Caminho para a pasta contendo os arquivos .ubyte do Fashion-MNIST.

    Retorna:
    X_train : np.ndarray, shape (60000, 784)
    y_train : np.ndarray, shape (60000,)
    X_test  : np.ndarray, shape (10000, 784)
    y_test  : np.ndarray, shape (10000,)
    """
    caminhos = {k: os.path.join(data_dir, v) for k, v in ARQUIVOS.items()}

    for nome, caminho in caminhos.items():
        if not os.path.exists(caminho):
            raise FileNotFoundError(
                f"Arquivo não encontrado: '{caminho}'\n"
                "Baixe os dados em:\n"
                "https://www.kaggle.com/datasets/zalando-research/fashionmnist\n"
                "e extraia os arquivos na pasta indicada em data_dir."
            )

    print("[loader] Carregando Fashion-MNIST...")
    X_train = _load_images(caminhos["train_images"])
    y_train = _load_labels(caminhos["train_labels"])
    X_test  = _load_images(caminhos["test_images"])
    y_test  = _load_labels(caminhos["test_labels"])

    print(f"[loader] Treino : X={X_train.shape}, y={y_train.shape}")
    print(f"[loader] Teste  : X={X_test.shape},  y={y_test.shape}")

    return X_train, y_train, X_test, y_test


def preprocessar(X_train, X_test):
    """
    Normaliza os pixels para o intervalo [0, 1] dividindo por 255.

    Decisão de projeto:
    - Normalização aplicada apenas com base no conjunto de treino (255.0 fixo),
      evitando data leakage.
    - Mantém consistência entre todas as abordagens (Baseline, Wrapper, GA).

    Retorna:
    X_train_norm : np.ndarray, dtype float32
    X_test_norm  : np.ndarray, dtype float32
    """
    X_train_norm = X_train.astype(np.float32) / 255.0
    X_test_norm  = X_test.astype(np.float32)  / 255.0

    print(f"[loader] Pré-processamento concluído. Dtype: {X_train_norm.dtype}")
    return X_train_norm, X_test_norm
