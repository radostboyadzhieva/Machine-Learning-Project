import numpy as np

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
                mse = np.mean((y_val - preds) ** 2)
                mse_list.append(mse)

            avg_mse = np.mean(mse_list)
            print(f"depth={max_depth}, split={min_split} -> avg MSE: {avg_mse:.4f}")

            if avg_mse < best_mse:
                best_mse = avg_mse
                best_params = (max_depth, min_split)

    print("\nBest parameters:", best_params)
    print("Best CV average MSE:", best_mse)
    return best_params, best_mse