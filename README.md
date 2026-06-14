## Dependências

```bash
pip install scikit-learn numpy
```

## Dados

Baixe o dataset Fashion-MNIST em:
https://www.kaggle.com/datasets/zalando-research/fashionmnist

Extraia os 4 arquivos `.ubyte` na raiz do projeto (ou ajuste `DATA_DIR` em cada `main_*.py`):
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

## Como executar

### Integrante 1 — Baseline
```bash
python main_baseline.py
```

### Integrante 2 — Wrapper
```bash
python main_wrapper.py
```

### Integrante 3 — Algoritmo Genético
```bash
python main_ga.py
```

### Integrante 4 — Comparação final
```bash
python main_comparacao.py
```

## Módulos compartilhados

| Módulo | Responsável | Reutilizado por |
|---|---|---|
| `data/loader.py` | Integrante 1 | Todos |
| `results/exporter.py` | Integrante 1 | Todos |
| `models/baseline.py` | Integrante 1 | Integrante 4 |

## Observações importantes

- A seleção de features (Wrapper e GA) deve usar **apenas os dados de treino**.
- A acurácia final de todos os modelos é calculada nos **dados de teste**.
- O modelo usado em todas as abordagens é `DecisionTreeClassifier(random_state=1)`.
