import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def clean_data(file):
    df = pd.read_csv(file)

    Y = df['ClaimNb']

    df['VehIsregular'] = (df['VehGas'] == 'Regular').astype(int)

    df = df.drop(columns= ['VehGas', 'IDpol', 'ClaimNb'])

    def letter_to_index(letter):
        return ord(letter.upper()) - ord('A')

    df['Area'] = df['Area'].apply(letter_to_index)

    df['Exposure'] = df['Exposure'].apply(lambda x: x - 1 if x > 1 else x)

    df = pd.get_dummies(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    X = pd.DataFrame(X_scaled, columns=df.columns)

    
    return X, Y

def split_data(X, Y, test_size):
    return train_test_split(X, Y, test_size=test_size, random_state=1)

def best_split(X, y):
    n_features = X.shape[1]
    best_rss = float('inf')
    best_feature = None
    best_threshold = None
    best_X_left = None
    best_y_left = None
    best_X_right = None
    best_y_right = None

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_mask = X[:, feature_index] < threshold
            right_mask = ~left_mask
            X_left, y_left, X_right, y_right = X[left_mask], y[left_mask], X[right_mask], y[right_mask]
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            rss = np.sum((y_left - np.mean(y_left))**2) + np.sum((y_right - np.mean(y_right))**2)

            if rss < best_rss:
                best_rss = rss
                best_feature = feature_index
                best_threshold = threshold
                best_X_left = X_left
                best_y_left = y_left
                best_X_right = X_right
                best_y_right = y_right

    return best_feature, best_threshold, best_X_left, best_y_left, best_X_right, best_y_right

def build_tree(X, y, max_depth, min_samples_split, depth=0):
    if len(y) < min_samples_split or depth >= max_depth or np.all(y == y[0]):
        return np.mean(y)

    best_feature, best_threshold, X_left, y_left, X_right, y_right = best_split(X, y)

    return {
        'feature': best_feature,
        'threshold': best_threshold,
        'left': build_tree(X_left, y_left, max_depth, min_samples_split, depth + 1),
        'right': build_tree(X_right, y_right, max_depth, min_samples_split, depth + 1)
    }

def predict_one(sample, tree):
    if not isinstance(tree, dict):
        return tree

    feature = tree['feature']
    threshold = tree['threshold']

    if sample[feature] < threshold:
        return predict_one(sample, tree['left'])
    else:
        return predict_one(sample, tree['right'])

def predict(X, tree):
    return np.array([predict_one(sample, tree) for sample in X])

def tune_tree(X, y, max_depth_list, min_samples_split_list, k=3):
    np.random.seed(1)

    best_mse = float('inf')
    best_params = None

    for max_depth in max_depth_list:
        for min_split in min_samples_split_list:
            indices = np.random.permutation(len(X))
            folds = np.array_split(indices, k)
            mse_list = []

            for i in range(k):
                val_idx = folds[i]
                train_idx = np.hstack([folds[j] for j in range(k) if j != i])

                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                
                tree = build_tree(X_train, y_train, max_depth=max_depth, min_samples_split=min_split)
                preds = predict(X_val, tree)
                mse = mean_squared_error(y_val, preds)
                mse_list.append(mse)

            avg_mse = np.mean(mse_list)
            print(f"depth={max_depth}, split={min_split} -> avg MSE: {avg_mse:.4f}")

            if avg_mse < best_mse:
                best_mse = avg_mse
                best_params = (max_depth, min_split)

    print("\nBest parameters:", best_params)
    print("Best CV average MSE:", best_mse)
    return best_params, best_mse


# Setup
out_folder = "rf_outputs"
os.makedirs(out_folder, exist_ok=True)

pink = "#F06292"
edge = "#4A4A4A"
plt.rcParams.update({'font.size': 11})

# Load trained model
rf = joblib.load("rf_final_model.pkl")
print("Random Forest model loaded.")

# Load data ONLY to get feature names
X, _ = clean_data("claims_train.csv")

# Safety check
if not isinstance(X, pd.DataFrame):
    raise TypeError("clean_data must return a pandas DataFrame to get feature names")

# Feature importances
importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

top10 = importances.head(10)

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
top10.plot(
    kind="bar",
    ax=ax,
    color=pink,
    edgecolor=edge
)

ax.set_title("Top 10 Feature Importances – Random Forest")
ax.set_ylabel("Importance")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.grid(True, alpha=0.25)

# Save
plt.tight_layout()
plt.savefig(
    os.path.join(out_folder, "RF_top10_feature_importances.png"),
    dpi=300
)
plt.close()

print("Top 10 feature importance plot saved.")


