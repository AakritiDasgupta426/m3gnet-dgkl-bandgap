import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import gpytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sys.path.append("/depot/amannodi/data/2026_Spring_UG/M3GNet-DGKL/DGKL")
from cat_uncertainty.dgkl.dgkl import DGKL

# ----------------------------
# Load data
# ----------------------------
data_path = "/depot/amannodi/data/2026_Spring_UG/M3GNet_embeds/Vectors_HaP_mf.pt"
data = torch.load(data_path, map_location="cpu")

print("Total samples:", len(data))
print("Fidelity counts:", Counter(data[i]["Fidelity"] for i in data))

valid_ids = [
    i for i in data
    if data[i]["Fidelity"] in ["PBE", "HSE06-PBE+SOC"]
]

base_X = np.array([data[i]["Vector"] for i in valid_ids], dtype=np.float32)

fidelity_vectors = []
for i in valid_ids:
    if data[i]["Fidelity"] == "PBE":
        fidelity_vectors.append([1.0, 0.0])
    else:
        fidelity_vectors.append([0.0, 1.0])

fidelity_vectors = np.array(fidelity_vectors, dtype=np.float32)

X = np.concatenate([base_X, fidelity_vectors], axis=1)
y = np.array([data[i]["Band_gap"] for i in valid_ids], dtype=np.float32)
compositions = np.array([data[i]["Composition"] for i in valid_ids])
fidelities = np.array([data[i]["Fidelity"] for i in valid_ids])

print("Multi-fidelity X shape:", X.shape)
print("y shape:", y.shape)
print("Fidelity counts:", Counter(fidelities))
# ----------------------------
# Split data, preserving composition alignment
# ----------------------------
X_train, X_temp, y_train, y_temp, comp_train, comp_temp = train_test_split(
    X, y, compositions, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test, comp_val, comp_test = train_test_split(
    X_temp, y_temp, comp_temp, test_size=0.5, random_state=42
)

print("Train shape:", X_train.shape, y_train.shape)
print("Val shape:", X_val.shape, y_val.shape)
print("Test shape:", X_test.shape, y_test.shape)

# ----------------------------
# Normalize using TRAIN statistics only
# ----------------------------
X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True)
X_std[X_std < 1e-8] = 1.0

X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

y_mean = y_train.mean()
y_std = y_train.std()
if y_std < 1e-8:
    y_std = 1.0

# Save original targets for metric computation in physical units
y_train_orig = y_train.copy()
y_val_orig = y_val.copy()
y_test_orig = y_test.copy()

# Normalize targets for training
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

print("X normalization done")
print("y mean", float(y_mean), "y std", float(y_std))

# ----------------------------
# Convert to tensors
# ----------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1)

y_train_orig = torch.tensor(y_train_orig, dtype=torch.float32).view(-1)
y_val_orig = torch.tensor(y_val_orig, dtype=torch.float32).view(-1)
y_test_orig = torch.tensor(y_test_orig, dtype=torch.float32).view(-1)

# ----------------------------
# Model
# ----------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=132, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_train = X_train.to(device)
X_val = X_val.to(device)
X_test = X_test.to(device)

y_train = y_train.to(device)
y_val = y_val.to(device)
y_test = y_test.to(device)

y_train_orig = y_train_orig.to(device)
y_val_orig = y_val_orig.to(device)
y_test_orig = y_test_orig.to(device)

feature_extractor = FeatureExtractor(input_dim=132, hidden_dim=64, latent_dim=32).to(device)

with torch.no_grad():
    Z_init = feature_extractor(X_train)

num_inducing = min(128, Z_init.shape[0])
perm = torch.randperm(Z_init.shape[0], device=Z_init.device)[:num_inducing]
inducing_points = Z_init[perm].clone()

model = DGKL(
    inducing_points=inducing_points,
    feature_extractor=feature_extractor,
    kernel_type="rbf",
    dist_type="cholesky",
    variational_strategy="standard",
).to(device)

likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model.gp.likelihood = likelihood

print("Initial latent shape:", Z_init.shape)
print("Inducing points shape:", inducing_points.shape)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
mll = gpytorch.mlls.VariationalELBO(
    likelihood,
    model.gp,
    num_data=X_train.size(0),
)

# ----------------------------
# Training with checkpointing
# ----------------------------
num_epochs = 200
patience = 5  # measured in validation checks, not raw epochs

best_val_rmse = float("inf")
best_epoch = -1
epochs_no_improve = 0

train_losses = []
val_rmses = []
val_maes = []
eval_epochs = []

model.train()
likelihood.train()

for epoch in range(num_epochs):
    optimizer.zero_grad()

    output = model((X_train,))
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    if epoch % 10 == 0:
        model.eval()
        likelihood.eval()

        with torch.no_grad():
            val_dist = likelihood(model((X_val,)))
            val_pred_norm = val_dist.mean
            val_std_norm = val_dist.stddev

            # Unnormalize to physical units
            val_pred = val_pred_norm * y_std + y_mean
            val_std = val_std_norm * y_std

            val_true = y_val_orig
            val_rmse = torch.sqrt(torch.mean((val_pred - val_true) ** 2)).item()
            val_mae = torch.mean(torch.abs(val_pred - val_true)).item()

        val_rmses.append(val_rmse)
        val_maes.append(val_mae)
        eval_epochs.append(epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "likelihood_state_dict": likelihood.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_rmse": best_val_rmse,
                },
                "best_dgkl_checkpoint.pt",
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        model.train()
        likelihood.train()

# ----------------------------
# Load best checkpoint
# ----------------------------
ckpt = torch.load("best_dgkl_checkpoint.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
likelihood.load_state_dict(ckpt["likelihood_state_dict"])

print(f"\nLoaded best checkpoint from epoch {ckpt['epoch']} with val RMSE {ckpt['best_val_rmse']:.4f}")

# ----------------------------
# Final evaluation on all splits
# ----------------------------
model.eval()
likelihood.eval()

with torch.no_grad():
    # Train
    train_dist = likelihood(model((X_train,)))
    y_train_pred_norm = train_dist.mean
    y_train_std_norm = train_dist.stddev
    y_train_pred = y_train_pred_norm * y_std + y_mean
    y_train_std = y_train_std_norm * y_std

    # Val
    val_dist = likelihood(model((X_val,)))
    y_val_pred_norm = val_dist.mean
    y_val_std_norm = val_dist.stddev
    y_val_pred = y_val_pred_norm * y_std + y_mean
    y_val_std = y_val_std_norm * y_std

    # Test
    test_dist = likelihood(model((X_test,)))
    y_test_pred_norm = test_dist.mean
    y_test_std_norm = test_dist.stddev
    y_test_pred = y_test_pred_norm * y_std + y_mean
    y_test_std = y_test_std_norm * y_std

# Metrics in original units
train_rmse = torch.sqrt(torch.mean((y_train_pred - y_train_orig) ** 2)).item()
train_mae = torch.mean(torch.abs(y_train_pred - y_train_orig)).item()
train_r2 = r2_score(y_train_orig.detach().cpu().numpy(), y_train_pred.detach().cpu().numpy())

val_rmse = torch.sqrt(torch.mean((y_val_pred - y_val_orig) ** 2)).item()
val_mae = torch.mean(torch.abs(y_val_pred - y_val_orig)).item()
val_r2 = r2_score(y_val_orig.detach().cpu().numpy(), y_val_pred.detach().cpu().numpy())

test_rmse = torch.sqrt(torch.mean((y_test_pred - y_test_orig) ** 2)).item()
test_mae = torch.mean(torch.abs(y_test_pred - y_test_orig)).item()
test_r2 = r2_score(y_test_orig.detach().cpu().numpy(), y_test_pred.detach().cpu().numpy())

print("\nFinal Metrics:")
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
print("First 10 test stddevs:", y_test_std[:10].detach().cpu().numpy())

# ----------------------------
# High-uncertainty analysis
# ----------------------------
test_abs_error = torch.abs(y_test_pred - y_test_orig).detach().cpu().numpy()
test_std_np = y_test_std.detach().cpu().numpy()
test_pred_np = y_test_pred.detach().cpu().numpy()
test_true_np = y_test_orig.detach().cpu().numpy()

top_idx = np.argsort(-test_std_np)[:10]

print("\nTop 10 highest-uncertainty test samples:")
for rank, idx in enumerate(top_idx, start=1):
    print(
        f"{rank:02d} | comp={comp_test[idx]} | "
        f"true={test_true_np[idx]:.4f} | pred={test_pred_np[idx]:.4f} | "
        f"std={test_std_np[idx]:.4f} | abs_err={test_abs_error[idx]:.4f}"
    )

# ----------------------------
# Save results
# ----------------------------
results = {
    "y_train_true": y_train_orig.detach().cpu(),
    "y_train_pred": y_train_pred.detach().cpu(),
    "y_train_std": y_train_std.detach().cpu(),
    "train_compositions": comp_train.tolist(),

    "y_val_true": y_val_orig.detach().cpu(),
    "y_val_pred": y_val_pred.detach().cpu(),
    "y_val_std": y_val_std.detach().cpu(),
    "val_compositions": comp_val.tolist(),

    "y_test_true": y_test_orig.detach().cpu(),
    "y_test_pred": y_test_pred.detach().cpu(),
    "y_test_std": y_test_std.detach().cpu(),
    "test_compositions": comp_test.tolist(),
    "test_residuals": (y_test_pred - y_test_orig).detach().cpu(),

    "train_losses": train_losses,
    "val_rmses": val_rmses,
    "val_maes": val_maes,
    "eval_epochs": eval_epochs,

    "x_mean": torch.tensor(X_mean),
    "x_std": torch.tensor(X_std),
    "y_mean": float(y_mean),
    "y_std_scalar": float(y_std),

    "best_epoch": best_epoch,
    "best_val_rmse": best_val_rmse,

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

torch.save(results, "m3gnet_dgkl_multifidelity_results.pt")
print("\nSaved results to m3gnet_dgkl_multifidelity_results.pt")
