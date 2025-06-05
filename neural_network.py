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

# 1. Carregar os dados
print("Carregando os dados:")
df = pd.read_csv("train.csv")

# 2. Separar features e target
correlations = df.corr(numeric_only=True)["SalePrice"]
selected_features = correlations[correlations > 0.3].index.drop("SalePrice")
X_selected = df[selected_features]
print(X_selected.head)

X = df[selected_features]
y = df["SalePrice"].values.reshape(-1, 1)

print("X:", X)
print("Y:", y)

# 3. Separar colunas numéricas e categóricas
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# 4. Pipeline para preprocessamento
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

# Normalizar target
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

# 5. Dividir em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_processed, y_scaled, test_size=0.2, random_state=42)

# Converter para tensores do PyTorch
X_train_tensor = torch.tensor(X_train.toarray() if hasattr(X_train, "toarray") else X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val.toarray() if hasattr(X_val, "toarray") else X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# 6. Definir a arquitetura da RNA
class HousePriceModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

model = HousePriceModel(X_train_tensor.shape[1])

# 7. Treinar a rede
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

# Treinamento
epochs = 10000
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = loss_fn(val_pred, y_val_tensor)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")

# Após o código de treinamento...

# 8. Carregar test.csv e processar com o mesmo pipeline
df_test = pd.read_csv("test.csv")
ids = df_test["Id"]

# Remover coluna Id
X_test = df_test[selected_features]


# Preprocessar com o mesmo pipeline treinado
X_test_processed = preprocessor.transform(X_test)

# Converter para tensor do PyTorch
X_test_tensor = torch.tensor(X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed, dtype=torch.float32)

# 9. Fazer previsões com o modelo treinado
model.eval()
with torch.no_grad():
    y_test_pred_scaled = model(X_test_tensor).numpy()

# Desnormalizar a saída (retornar à escala original de SalePrice)
y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)

# 10. Criar o arquivo de submissão
submission = pd.DataFrame({
    "Id": ids,
    "SalePrice": y_test_pred.flatten()
})
submission.to_csv("submission.csv", index=False)

print("Arquivo 'submission.csv' gerado com sucesso!")

# Avaliação com validação
val_pred_np = y_scaler.inverse_transform(val_pred.numpy())
y_val_np = y_scaler.inverse_transform(y_val_tensor.numpy())

mae = mean_absolute_error(y_val_np, val_pred_np)
rmse = np.sqrt(mean_squared_error(y_val_np, val_pred_np))
r2 = r2_score(y_val_np, val_pred_np)

print(f"📊 Avaliação final:\nMAE: R$ {mae:,.2f}\nRMSE: R$ {rmse:,.2f}\nR²: {r2:.4f}")
media_preco = df["SalePrice"].mean()
print(f"🏠 Média dos valores das casas: R$ {media_preco:,.2f}")
