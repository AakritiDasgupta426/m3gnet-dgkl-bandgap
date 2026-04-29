import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = torch.load("m3gnet_dgkl_multifidelity_results.pt", map_location="cpu")
print("Available keys:", data.keys())

y_train_true = data["y_train_true"].numpy()
y_train_pred = data["y_train_pred"].numpy()
y_train_std = data["y_train_std"].numpy()

y_val_true = data["y_val_true"].numpy()
y_val_pred = data["y_val_pred"].numpy()
y_val_std = data["y_val_std"].numpy()

y_test_true = data["y_test_true"].numpy()
y_test_pred = data["y_test_pred"].numpy()
y_test_std = data["y_test_std"].numpy()

train_losses = np.array(data["train_losses"])
eval_epochs = np.array(data["eval_epochs"])


def summarize(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} RMSE: {rmse:.4f}")
    print(f"{name} MAE : {mae:.4f}")
    print(f"{name} R2  : {r2:.4f}")
    print()
    return rmse, mae, r2


train_rmse, train_mae, train_r2 = summarize(y_train_true, y_train_pred, "Train")
val_rmse, val_mae, val_r2 = summarize(y_val_true, y_val_pred, "Val")
test_rmse, test_mae, test_r2 = summarize(y_test_true, y_test_pred, "Test")

all_vals = np.concatenate([
    y_train_true, y_train_pred,
    y_val_true, y_val_pred,
    y_test_true, y_test_pred
])
mn, mx = all_vals.min(), all_vals.max()

# parity plot 
plt.figure(figsize=(7, 7))
plt.scatter(y_train_true, y_train_pred, alpha=0.35, label="Train")
plt.scatter(y_val_true, y_val_pred, alpha=0.7, label="Val")
plt.scatter(y_test_true, y_test_pred, alpha=0.85, label="Test")
plt.plot([mn, mx], [mn, mx], "--")
plt.xlabel("True Band Gap (eV)")
plt.ylabel("Predicted Band Gap (eV)")
plt.title("Parity Plot")
plt.legend()
plt.tight_layout()
plt.savefig("multifidelity_parity_plot.png", dpi=300)
plt.close()

# test residudals plot 
test_residuals = y_test_pred - y_test_true
plt.figure(figsize=(7, 5))
plt.scatter(y_test_true, test_residuals, alpha=0.85)
plt.axhline(0, linestyle="--")
plt.xlabel("True Band Gap (eV)")
plt.ylabel("Residual (Pred - True) (eV)")
plt.title("Test Residual Plot")
plt.tight_layout()
plt.savefig("multifidelity_residual_plot.png", dpi=300)
plt.close()

# the uncertaininty compared with avsolute error 
test_abs_error = np.abs(y_test_pred - y_test_true)
plt.figure(figsize=(7, 5))
plt.scatter(y_test_std, test_abs_error, alpha=0.85)
plt.xlabel("Predictive Std Dev (eV)")
plt.ylabel("Absolute Error (eV)")
plt.title("Uncertainty vs Absolute Error")
plt.tight_layout()
plt.savefig("multifidelity_uncertainty_vs_error.png", dpi=300)
plt.close()

# the test parity with some jazz 
plt.figure(figsize=(7, 6))
sc = plt.scatter(y_test_true, y_test_pred, c=y_test_std, alpha=0.9)
plt.plot([mn, mx], [mn, mx], "--")
plt.xlabel("True Band Gap (eV)")
plt.ylabel("Predicted Band Gap (eV)")
plt.title("Test Parity Plot Colored by Uncertainty")
plt.colorbar(sc, label="Predictive Std Dev (eV)")
plt.tight_layout()
plt.savefig("multifidelity_parity_uncertainty_colored.png", dpi=300)
plt.close()

# loss curve for trainign 
plt.figure(figsize=(7, 5))
plt.plot(range(len(train_losses)), train_losses)
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Training Loss Curve")
plt.tight_layout()
plt.savefig("multifidelity_train_loss_curve.png", dpi=300)
plt.close()

# historgrams regarding residuals 
plt.figure(figsize=(7, 5))
plt.hist(test_residuals, bins=20)
plt.xlabel("Residual (eV)")
plt.ylabel("Count")
plt.title("Test Residual Histogram")
plt.tight_layout()
plt.savefig("multifidelity_residual_histogram.png", dpi=300)
plt.close()

# histogram regarding uncertainiyu 
plt.figure(figsize=(7, 5))
plt.hist(y_test_std, bins=20)
plt.xlabel("Predictive Std Dev (eV)")
plt.ylabel("Count")
plt.title("Predictive Uncertainty Distribution")
plt.tight_layout()
plt.savefig("multifidelity_uncertainty_histogram.png", dpi=300)
plt.close()

# saving all of this in a summary for easy access 
with open("metrics_summary.txt", "w") as f:
    f.write(f"Train RMSE: {train_rmse:.4f}\n")
    f.write(f"Train MAE:  {train_mae:.4f}\n")
    f.write(f"Train R2:   {train_r2:.4f}\n\n")

    f.write(f"Val RMSE:   {val_rmse:.4f}\n")
    f.write(f"Val MAE:    {val_mae:.4f}\n")
    f.write(f"Val R2:     {val_r2:.4f}\n\n")

    f.write(f"Test RMSE:  {test_rmse:.4f}\n")
    f.write(f"Test MAE:   {test_mae:.4f}\n")
    f.write(f"Test R2:    {test_r2:.4f}\n")

#for confirmatin and debufging help
print("Saved:")
print(" - multifidelity_parity_plot.png")
print(" - multifidelity_residual_plot.png")
print(" - multifidelity_uncertainty_vs_error.png")
print(" - multifidelity_parity_uncertainty_colored.png")
print(" - multifidelity_train_loss_curve.png")
print(" - multifidelity_residual_histogram.png")
print(" - multifidelity_uncertainty_histogram.png")
print(" - multifidelity_metrics_summary.txt")
