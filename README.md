﻿# Trabalho de Redes Neurais Artificiais - Previsão de Preços de Casas

Este projeto utiliza uma rede neural construída em **PyTorch** para prever os preços de casas com base no conjunto de dados `train.csv` do desafio da Kaggle *House Prices: Advanced Regression Techniques*.

---

## Autores
- Brunno Aires Silva
- Lucas Hideki Abe
- Gianlluca do Carmo Leme

## Repositório
  https://github.com/hideki-abe/trabalho-rna/edit/master/README.md

## Instalação de Dependências

Abra o terminal e execute:

```bash
pip install torch pandas scikit-learn
```

---

## Execução do Código

Execute o script principal com:

```bash
python neural_network.py
```

---

## Saída Esperada (Exemplo)

```text
Early stopping at epoch 682
Arquivo 'submission.csv' gerado com sucesso!
Avaliação final:
MAE: R$ 15,360.11
RMSE: R$ 24,488.95
R²: 0.9218
Média dos valores das casas: R$ 180,921.20
RMSLE: 0.13272
```

---

## Arquivos Importantes

- `neural_network.py`: código principal da rede neural.
- `train.csv` / `test.csv`: arquivos de entrada (devem estar na mesma pasta).
- `submission.csv`: arquivo gerado com as previsões finais.

---

## 📌 Observações

- O target (`SalePrice`) é transformado com `log1p` para suavizar a distribuição.
- A avaliação usa MAE, RMSE, R² e RMSLE como métricas.
- Técnicas como feature engineering, dropout, batch normalization e `ReduceLROnPlateau` são utilizadas.


---
