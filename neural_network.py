# ===================================================
# Trabalho - Redes Neurais Artificiais (RNA)
# Inteligência Computacional - INF0092
# Objetivo: Previsão de preços de casas
# ===================================================

# === 1. Imports e configurações gerais ===
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configurar saída UTF-8 (caso necessário para prints no terminal)
import sys
sys.stdout.reconfigure(encoding='utf-8')

# === 2. Carregamento e pré-processamento dos dados ===
print("📥 Carregando os dados...")
df = pd.read_csv("train.csv")

# === 2.1. Remoção de outliers ===
df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000))]
upper_price_limit = df["SalePrice"].quantile(0.995)
df = df[df["SalePrice"] < upper_price_limit]

# === 2.2. Engenharia de atributos manuais ===
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
df["Age"] = df["YrSold"] - df["YearBuilt"]
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)

# === 2.3. Separação entre features e target ===
X = df.drop(columns=["Id", "SalePrice"])
y = np.log1p(df["SalePrice"].values.reshape(-1, 1))  # log-transformação

# === 2.4. Pipeline de pré-processamento ===
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])
X_processed = preprocessor.fit_transform(X)

# === 2.5. Normalização do target ===
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

# === 2.6. Divisão dos dados ===
X_train, X_val, y_train, y_val = train_test_split(X_processed, y_scaled, test_size=0.2, random_state=42)

# === 2.7. Conversão para tensores ===
to_tensor = lambda x: torch.tensor(x.toarray() if hasattr(x, "toarray") else x, dtype=torch.float32)
X_train_tensor = to_tensor(X_train)
X_val_tensor = to_tensor(X_val)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# === 3. Definição do modelo ===
class HousePriceModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.Dropout(0.3),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.BatchNorm1d(48),
            nn.Dropout(0.2),
            nn.Linear(48, 1)
        )

    def forward(self, x):
        return self.network(x)

model = HousePriceModel(X_train_tensor.shape[1])

# === 4. Treinamento ===
loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.5)

best_val_loss = float('inf')
patience = 2000
patience_counter = 0
best_model_state = None
epochs = 10000

# Capturar log de loss
train_loss_log = []
val_loss_log = []

print("🏋️ Iniciando treinamento...")
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor)
        val_loss = loss_fn(val_pred, y_val_tensor)

    train_loss_log.append(loss.item())
    val_loss_log.append(val_loss.item())

    scheduler.step(val_loss.item())

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")

model.load_state_dict(best_model_state)

# Salvando o .csv que contém os dados de loss
output_path = r"\loss_dados"

import os
os.makedirs(output_path, exist_ok=True)

df_log = pd.DataFrame({
    "epoch": list(range(1, len(train_loss_log)+1)),
    "train_loss": train_loss_log,
    "val_loss": val_loss_log
})
df_log.to_csv(os.path.join(output_path, "log_neural_network.csv"), index=False)
print(f"📁 Log de perda salvo em: {output_path}")

# === 5. Previsão no conjunto de teste ===
print("📈 Gerando submissão...")
df_test = pd.read_csv("test.csv")

# Repetir a engenharia de atributos
df_test["TotalSF"] = df_test["TotalBsmtSF"] + df_test["1stFlrSF"] + df_test["2ndFlrSF"]
df_test["TotalBath"] = df_test["FullBath"] + 0.5 * df_test["HalfBath"] + df_test["BsmtFullBath"] + 0.5 * df_test["BsmtHalfBath"]
df_test["Age"] = df_test["YrSold"] - df_test["YearBuilt"]
df_test["RemodAge"] = df_test["YrSold"] - df_test["YearRemodAdd"]
df_test["IsRemodeled"] = (df_test["YearBuilt"] != df_test["YearRemodAdd"]).astype(int)
df_test["HasGarage"] = (df_test["GarageArea"] > 0).astype(int)
df_test["HasBasement"] = (df_test["TotalBsmtSF"] > 0).astype(int)

ids = df_test["Id"]
X_test = df_test.drop(columns=["Id"])
X_test_processed = preprocessor.transform(X_test)
X_test_tensor = to_tensor(X_test_processed)

model.eval()
with torch.no_grad():
    y_test_pred_scaled = model(X_test_tensor).numpy()

y_test_pred = np.expm1(y_scaler.inverse_transform(y_test_pred_scaled))

submission = pd.DataFrame({
    "Id": ids,
    "SalePrice": y_test_pred.flatten()
})
submission.to_csv("submission.csv", index=False)
print("✅ Arquivo 'submission.csv' gerado com sucesso!")

# === 6. Avaliação final ===
val_pred_np = np.expm1(y_scaler.inverse_transform(val_pred.numpy()))
y_val_np = np.expm1(y_scaler.inverse_transform(y_val_tensor.numpy()))

mae = mean_absolute_error(y_val_np, val_pred_np)
rmse = np.sqrt(mean_squared_error(y_val_np, val_pred_np))
r2 = r2_score(y_val_np, val_pred_np)

def rmsle(y_true, y_pred):
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

rmsle_score = rmsle(y_val_np, val_pred_np)
media_preco = df["SalePrice"].mean()

print(f"📊 Avaliação final:")
print(f"MAE: R$ {mae:,.2f}")
print(f"RMSE: R$ {rmse:,.2f}")
print(f"R²: {r2:.4f}")
print(f"🏠 Média dos valores das casas: R$ {media_preco:,.2f}")
print(f"🧮 RMSLE: {rmsle_score:.5f}")