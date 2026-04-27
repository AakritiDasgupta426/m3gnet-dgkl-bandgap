import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

data_path = "/depot/amannodi/data/2026_Spring_UG/M3GNet_embeds/Vectors_HaP_mf.pt"
data = torch.load(data_path, map_location="cpu")

print("Total samples:", len(data))
print("Fidelity counts:", Counter(data[i]["Fidelity"] for i in data))

pbe_ids = [i for i in data if data[i]["Fidelity"] == "PBE"]

X = np.array([data[i]["Vector"] for i in pbe_ids], dtype=np.float32)
y = np.array([data[i]["Band_gap"] for i in pbe_ids], dtype=np.float32)
compositions = np.array([data[i]["Composition"] for i in pbe_ids])

X_train, X_temp, y_train, y_temp, comp_train, comp_temp = train_test_split(
    X, y, compositions, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test, comp_val, comp_test = train_test_split(
    X_temp, y_temp, comp_temp, test_size=0.5, random_state=42
)

print("Train shape:", X_train.shape, y_train.shape)
print("Val shape:", X_val.shape, y_val.shape)
print("Test shape:", X_test.shape, y_test.shape)

# Normalize X using train stats
X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True)
X_std[X_std < 1e-8] = 1.0

X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Save original y
y_train_orig = y_train.copy()
y_val_orig = y_val.copy()
y_test_orig = y_test.copy()

# GPR baseline
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    normalize_y=True,
    n_restarts_optimizer=2,
    random_state=42
)

print("Fitting GPR...")
gpr.fit(X_train, y_train)

# Predictions
y_train_pred, y_train_std = gpr.predict(X_train, return_std=True)
y_val_pred, y_val_std = gpr.predict(X_val, return_std=True)
y_test_pred, y_test_std = gpr.predict(X_test, return_std=True)

# Metrics
train_rmse = np.sqrt(np.mean((y_train_pred - y_train_orig) ** 2))
train_mae = np.mean(np.abs(y_train_pred - y_train_orig))
train_r2 = r2_score(y_train_orig, y_train_pred)

val_rmse = np.sqrt(np.mean((y_val_pred - y_val_orig) ** 2))
val_mae = np.mean(np.abs(y_val_pred - y_val_orig))
val_r2 = r2_score(y_val_orig, y_val_pred)

test_rmse = np.sqrt(np.mean((y_test_pred - y_test_orig) ** 2))
test_mae = np.mean(np.abs(y_test_pred - y_test_orig))
test_r2 = r2_score(y_test_orig, y_test_pred)

print("\nFinal GPR Metrics:")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Train MAE:  {train_mae:.4f}")
print(f"Train R2:   {train_r2:.4f}")
print(f"Val RMSE:   {val_rmse:.4f}")
print(f"Val MAE:    {val_mae:.4f}")
print(f"Val R2:     {val_r2:.4f}")
print(f"Test RMSE:  {test_rmse:.4f}")
print(f"Test MAE:   {test_mae:.4f}")
print(f"Test R2:    {test_r2:.4f}")

print("\nSample predictive uncertainties:")
print("First 10 test stddevs:", y_test_std[:10])

# Top uncertainty points
test_abs_error = np.abs(y_test_pred - y_test_orig)
top_idx = np.argsort(-y_test_std)[:10]

print("\nTop 10 highest-uncertainty test samples:")
for rank, idx in enumerate(top_idx, start=1):
    print(
        f"{rank:02d} | comp={comp_test[idx]} | "
        f"true={y_test_orig[idx]:.4f} | pred={y_test_pred[idx]:.4f} | "
        f"std={y_test_std[idx]:.4f} | abs_err={test_abs_error[idx]:.4f}"
    )

results = {
    "y_train_true": y_train_orig,
    "y_train_pred": y_train_pred,
    "y_train_std": y_train_std,
    "train_compositions": comp_train.tolist(),

    "y_val_true": y_val_orig,
    "y_val_pred": y_val_pred,
    "y_val_std": y_val_std,
    "val_compositions": comp_val.tolist(),

    "y_test_true": y_test_orig,
    "y_test_pred": y_test_pred,
    "y_test_std": y_test_std,
    "test_compositions": comp_test.tolist(),
    "test_residuals": y_test_pred - y_test_orig,

    "x_mean": X_mean,
    "x_std": X_std,

    "train_rmse": train_rmse,
    "train_mae": train_mae,
    "train_r2": train_r2,
    "val_rmse": val_rmse,
    "val_mae": val_mae,
    "val_r2": val_r2,
    "test_rmse": test_rmse,
    "test_mae": test_mae,
    "test_r2": test_r2,
}

torch.save(results, "gpr_baseline_results.pt")
print("\nSaved results to gpr_baseline_results.pt")
