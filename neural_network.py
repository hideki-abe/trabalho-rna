
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

# 1. Carregamento
df = pd.read_csv("train.csv")
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
df["Age"] = df["YrSold"] - df["YearBuilt"]
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
df["GarageScore"] = df["GarageArea"] * df["GarageCars"]
df["TotalPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
df["TotalHouseArea"] = df["GrLivArea"] + df["TotalBsmtSF"]
df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

neighborhood_price = df.groupby("Neighborhood")["SalePrice"].mean()
df["NeighborhoodEncoded"] = df["Neighborhood"].map(neighborhood_price)

X = df.drop(columns=["Id", "SalePrice"])
y = np.log1p(df["SalePrice"].values.reshape(-1, 1))

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

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
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X_processed, y_scaled, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train.toarray() if hasattr(X_train, "toarray") else X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.toarray() if hasattr(X_val, "toarray") else X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

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

def train_model(optimizer_name):
    model = HousePriceModel(X_train_tensor.shape[1])
    loss_fn = nn.SmoothL1Loss()
    optimizer = {
        'adamw': torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4),
        'rmsprop': torch.optim.RMSprop(model.parameters(), lr=0.0005, weight_decay=1e-4)
    }[optimizer_name]
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.5)
    best_model_state = None
    best_val_loss = float('inf')
    patience = 500
    patience_counter = 0

    for epoch in range(10000):
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
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/10000 - Train Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")
    model.load_state_dict(best_model_state)
    return model

model_adamw = train_model("adamw")
model_rmsprop = train_model("rmsprop")

model_adamw.eval()
model_rmsprop.eval()
with torch.no_grad():
    val_pred_adamw = model_adamw(X_val_tensor).numpy()
    val_pred_rmsprop = model_rmsprop(X_val_tensor).numpy()
    val_pred_avg = (val_pred_adamw + val_pred_rmsprop) / 2

val_pred_np = np.expm1(y_scaler.inverse_transform(val_pred_avg))
y_val_np = np.expm1(y_scaler.inverse_transform(y_val_tensor.numpy()))

mae = mean_absolute_error(y_val_np, val_pred_np)
rmse = np.sqrt(mean_squared_error(y_val_np, val_pred_np))
r2 = r2_score(y_val_np, val_pred_np)

def rmsle(y_true, y_pred):
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

rmsle_score = rmsle(y_val_np, val_pred_np)
print(f"MAE: R$ {mae:,.2f}")
print(f"RMSE: R$ {rmse:,.2f}")
print(f"R²: {r2:.4f}")
print(f"RMSLE: {rmsle_score:.5f}")
