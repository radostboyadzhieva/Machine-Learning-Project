import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from functions import clean_data, split_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error


out_folder = "rf_outputs"
os.makedirs(out_folder, exist_ok=True)


pink = "#F06292"
edge = "#4A4A4A"
plt.rcParams.update({'font.size': 11})


random_state = 42
cv_splits = 3

def mse(model, X, y):
    preds = model.predict(X)
    return mean_squared_error(y, preds)


X_full, Y_full = clean_data("claims_train.csv")
X = X_full
Y = Y_full.values if hasattr(Y_full, 'values') else np.array(Y_full)
print(f"Using full dataset: {X.shape[0]} rows")


X_train, X_valid, y_train, y_valid = split_data(X, Y, test_size=0.20)


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_leaf': [50, 75, 100, 125],
    'max_features': ['sqrt', 0.5, 0.7]
}

cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
rf_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)

grid = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=cv,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

print(f"Running GridSearchCV ({cv_splits}-fold CV)...")
grid.fit(X_train, y_train)

best_params = grid.best_params_
best_cv_mse = -grid.best_score_


best_rf = RandomForestRegressor(
    **best_params,
    random_state=random_state,
    n_jobs=-1
)
best_rf.fit(X_train, y_train)
mse_rf_best_val = mse(best_rf, X_valid, y_valid)


print("MODEL PERFORMANCE SUMMARY")
print(f"Best RF hyperparameters : {best_params}")
print(f"CV mean MSE (train folds): {best_cv_mse:.6f}")

# Plot 1: Predicted vs Actual
y_pred_best = best_rf.predict(X_valid)
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(y_valid, y_pred_best, alpha=0.25, color=pink, edgecolors='none')
mx = max(y_valid.max(), y_pred_best.max())
ax.plot([0, mx], [0, mx], color=edge, linestyle='--', linewidth=1)
ax.set_xlabel("Actual ClaimNb")
ax.set_ylabel("Predicted ClaimNb")
ax.set_title("Best Random Forest: Predicted vs Actual")
ax.grid(True, alpha=0.25)
fig.savefig(os.path.join(out_folder, "RF_pred_vs_actual.png"), dpi=300)
plt.show()
plt.close(fig)

# Plot 2: Feature Importances
fi = best_rf.feature_importances_
feat_names = [f"f{i}" for i in range(len(fi))]
importances = pd.Series(fi, index=feat_names)
top20 = importances.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10,6))
top20.plot(kind='bar', ax=ax, color=pink, edgecolor=edge)
ax.set_title("Best RF – Top 20 Feature Importances")
ax.set_ylabel("Importance")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.grid(True, alpha=0.25)
fig.savefig(os.path.join(out_folder, "RF_feature_importances.png"), dpi=300)
plt.show()
plt.close(fig)

# Plot 3: ResidualsHistogram
residuals = y_valid - y_pred_best

fig, ax = plt.subplots(figsize=(7,5))
ax.hist(residuals, bins=40, color=pink, edgecolor=edge)
ax.set_title("Residuals (Actual – Predicted)")
ax.set_xlabel("Residual")
ax.set_ylabel("Count")
ax.grid(True, alpha=0.25)
fig.savefig(os.path.join(out_folder, "RF_residuals.png"), dpi=300)
plt.show()
plt.close(fig)


final_rf = RandomForestRegressor(
    **best_params,
    random_state=random_state,
    n_jobs=-1
)
final_rf.fit(X, Y)
print("Final RF model trained on the full dataset.")

joblib.dump(final_rf, "rf_final_model.pkl")
print("Final RF model saved to rf_final_model.pkl")
