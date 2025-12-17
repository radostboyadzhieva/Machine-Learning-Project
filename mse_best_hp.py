import numpy as np
from functions import clean_data, split_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Fixed best hyperparameters

best_params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 50,
    "max_features": 0.7
}

random_state = 42

# Load TRAIN data

X, y = clean_data("claims_train.csv")
y = y.values if hasattr(y, "values") else np.array(y)

# Train / validation split
X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.20)

# Train model

rf = RandomForestRegressor(
    **best_params,
    random_state=random_state,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# MSEs

train_mse = mean_squared_error(y_train, rf.predict(X_train))
val_mse   = mean_squared_error(y_val, rf.predict(X_val))

# Load TEST data

X_test, y_test = clean_data("claims_test.csv")
y_test = y_test.values if hasattr(y_test, "values") else np.array(y_test)

test_mse = mean_squared_error(y_test, rf.predict(X_test))


print("Random Forest MSE results")
print(f"Training MSE:    {train_mse:.6f}")
print(f"Validation MSE:  {val_mse:.6f}")
print(f"Test MSE:        {test_mse:.6f}")

