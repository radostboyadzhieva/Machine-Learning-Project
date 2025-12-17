import os
import numpy as np
import matplotlib.pyplot as plt
from functions import clean_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Setup

out_folder = "rf_outputs"
os.makedirs(out_folder, exist_ok=True)

pink = "#F06292"
edge = "#4A4A4A"
plt.rcParams.update({'font.size': 11})

random_state = 42

# Best hyperparameters

best_params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 50,
    "max_features": 0.7
}

# Train model on FULL TRAIN data

X_train, y_train = clean_data("claims_train.csv")
y_train = y_train.values if hasattr(y_train, "values") else np.array(y_train)

rf = RandomForestRegressor(
    **best_params,
    random_state=random_state,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Load TEST data

X_test, y_test = clean_data("claims_test.csv")
y_test = y_test.values if hasattr(y_test, "values") else np.array(y_test)


# Predict on TEST data

y_pred_test = rf.predict(X_test)

test_mse = mean_squared_error(y_test, y_pred_test)
print(f"Test MSE: {test_mse:.6f}")

# Plot: Predicted vs Actual (TEST)

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(
    y_test,
    y_pred_test,
    alpha=0.25,
    color=pink,
    edgecolors="none"
)

mx = max(y_test.max(), y_pred_test.max())
ax.plot([0, mx], [0, mx], linestyle="--", linewidth=1, color=edge)

ax.set_xlabel("Actual ClaimNb (Test)")
ax.set_ylabel("Predicted ClaimNb (Test)")
ax.set_title("Random Forest – Predicted vs Actual (Test Data)")
ax.grid(True, alpha=0.25)

fig.savefig(
    os.path.join(out_folder, "RF_test_pred_vs_actual.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

