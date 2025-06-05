import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 1. Carregar os dados
print("Carregando os dados:")
df = pd.read_csv("train.csv")

# 2. Usar todas as features (menos Id e SalePrice)
X = df.drop(columns=["Id", "SalePrice"])
y = df["SalePrice"].values.reshape(-1, 1)

# 3. Separar colunas numéricas e categóricas
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# 4. Pipeline de preprocessamento
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

# 5. Normalizar target
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

# 6. Dividir em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_processed, y_scaled, test_size=0.2, random_state=42)

# 7. Converter para tensores do PyTorch
X_train_tensor = torch.tensor(X_train.toarray() if hasattr(X_train, "toarray") else X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val.toarray() if hasattr(X_val, "toarray") else X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# 8. Arquitetura da rede
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

# 9. Treinamento
loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.5)

best_val_loss = float('inf')
patience = 300
patience_counter = 0
best_model_state = None

epochs = 10000
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

# 10. Test set
df_test = pd.read_csv("test.csv")
ids = df_test["Id"]
X_test = df_test.drop(columns=["Id"])
X_test_processed = preprocessor.transform(X_test)

X_test_tensor = torch.tensor(X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed, dtype=torch.float32)

# 11. Previsões
model.eval()
with torch.no_grad():
    y_test_pred_scaled = model(X_test_tensor).numpy()

y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)

# 12. Submissão
submission = pd.DataFrame({
    "Id": ids,
    "SalePrice": y_test_pred.flatten()
})
submission.to_csv("submission.csv", index=False)

print("✅ Arquivo 'submission.csv' gerado com sucesso!")

# 13. Avaliação
val_pred_np = y_scaler.inverse_transform(val_pred.numpy())
y_val_np = y_scaler.inverse_transform(y_val_tensor.numpy())

mae = mean_absolute_error(y_val_np, val_pred_np)
rmse = np.sqrt(mean_squared_error(y_val_np, val_pred_np))
r2 = r2_score(y_val_np, val_pred_np)
# 13. Calcular RMSLE
def rmsle(y_true, y_pred):
    # Garantir que não haja valores negativos
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

rmsle_score = rmsle(y_val_np, val_pred_np)



media_preco = df["SalePrice"].mean()

print(f"📊 Avaliação final:\nMAE: R$ {mae:,.2f}\nRMSE: R$ {rmse:,.2f}\nR²: {r2:.4f}")
print(f"🏠 Média dos valores das casas: R$ {media_preco:,.2f}")
print(f"🧮 RMSLE: {rmsle_score:.5f}")
