import numpy as np
from functions import clean_data, split_data

def initialize_parameters(input_layer, hidden_layer1_neurons, hidden_layer2_neurons, output_layer):
    np.random.seed(42)
    parameters = {

    "w1": np.random.randn(input_layer, hidden_layer1_neurons) * np.sqrt(2.0 / input_layer),
    "b1": np.zeros((1, hidden_layer1_neurons)),

    "w2":np.random.randn(hidden_layer1_neurons, hidden_layer2_neurons) * np.sqrt(2.0 / hidden_layer1_neurons),
    "b2": np.zeros((1, hidden_layer2_neurons)), 

    "w3": np.random.randn(hidden_layer2_neurons,output_layer) * np.sqrt(2.0 / hidden_layer2_neurons),
    "b3": np.zeros((1, output_layer))}
    
    return parameters

def relu(z):
    return np.maximum(0, z)

def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

def relu_derivative(z):
    return (z > 0).astype(float)

def softplus_derivative(z):
    return 1.0 / (1.0 + np.exp(-z))

def get_output_derivative(output_function):
    if output_function is softplus:
        return softplus_derivative
    elif output_function is relu:
        return relu_derivative

def feed_forward(X, parameters, output_function):
    w1, b1 = parameters["w1"], parameters["b1"]
    w2, b2 = parameters["w2"], parameters["b2"]
    w3, b3 = parameters["w3"], parameters["b3"]
    
    z1 = X @ w1 + b1
    a1 = relu(z1)
    
    z2 = a1 @ w2 + b2
    a2 = relu(z2)
    
    z3 = a2 @ w3 + b3
    y_pred = output_function(z3)
    
    values = {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "predicted_y": y_pred}
    return y_pred, values

def backward_propagation(X, y, parameters, values, output_func_der):
    w1, b1 = parameters["w1"], parameters["b1"]
    w2, b2 = parameters["w2"], parameters["b2"]
    w3, b3 = parameters["w3"], parameters["b3"]
    
    z1, a1 = values["z1"], values["a1"]
    z2, a2 = values["z2"], values["a2"]
    z3 = values["z3"]
    y_pred = values["predicted_y"]
    m = X.shape[0]
    
    y = y.reshape(-1, 1)
    dL_dlambda = 1 - y / (y_pred + 1e-8)
    dz3 = dL_dlambda * output_func_der(z3)    
    dw3 = (a2.T @ dz3) / m                       
    db3 = np.sum(dz3, axis=0, keepdims=True) / m  

    da2 = dz3 @ w3.T                              
    dz2 = da2 * relu_derivative(z2)               
    dw2 = (a1.T @ dz2) / m                       
    db2 = np.sum(dz2, axis=0, keepdims=True) / m  
    
    da1 = dz2 @ w2.T                              
    dz1 = da1 * relu_derivative(z1)              
    dw1 = (X.T @ dz1) / m                         
    db1 = np.sum(dz1, axis=0, keepdims=True) / m 
    
    gradients = {
        "w1": dw1,
        "b1": db1,
        "w2": dw2,
        "b2": db2,
        "w3": dw3,
        "b3": db3,
    }
    
    return gradients

def init_adam_state(parameters):
    adam_state = {
        "t": 0,
        "m": {},
        "v": {}
    }
    
    for name, value in parameters.items():
        adam_state["m"][name] = np.zeros_like(value)
        adam_state["v"][name] = np.zeros_like(value)
        
    return adam_state

def adam_update(parameters, grads, adam_state, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    adam_state["t"] += 1
    t = adam_state["t"]

    for name in parameters.keys():
        g = grads[name]

        adam_state["m"][name] = beta1 * adam_state["m"][name] + (1 - beta1) * g
        adam_state["v"][name] = beta2 * adam_state["v"][name] + (1 - beta2) * (g ** 2)

        m_hat = adam_state["m"][name] / (1 - beta1 ** t)
        v_hat = adam_state["v"][name] / (1 - beta2 ** t)

        parameters[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    return parameters, adam_state

def train_model(X_train, y_train, X_val, y_val, learning_rate, batch_size, epochs,output_function, output_func_der):
    parameters = initialize_parameters(
        input_layer = X_train.shape[1],
        hidden_layer1_neurons = 28,
        hidden_layer2_neurons = 28,
        output_layer = 1
    )
    
    adam_state = init_adam_state(parameters)
    best_val_loss = float('inf')
    best_val_mse_epoch = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        
        for i in range(0, len(X_shuffled), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size] 
            
            _, values = feed_forward(X_batch, parameters, output_function)
            
            gradients = backward_propagation(
                X_batch, y_batch, parameters, values, output_func_der
            )

            parameters, adam_state = adam_update(
                parameters, gradients, adam_state,
                learning_rate=learning_rate  
            )
        
        y_val_pred, _ = feed_forward(X_val, parameters,output_function)
        y_val_pred = y_val_pred.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        val_loss = np.mean(y_val_pred - y_val * np.log(y_val_pred + 1e-8))
        
        val_mse_epoch = np.mean((y_val - y_val_pred)**2)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mse_epoch = val_mse_epoch
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 5:   
            print(f"Early stopping at epoch {epoch}")
            break
        
    return best_val_loss, best_val_mse_epoch, best_epoch


def tune_nn(X, y, batch_sizes, output_functions, learning_rate=0.001, epochs=150,k=3):
    np.random.seed(42)
    
    best_avg_loss = float("inf")
    best_batch_size = None
    best_out_func = None
    best_epoch_final = None

    for batch_size in batch_sizes:
        for out_func, (output_function, output_func_der) in output_functions.items():
            
            indices = np.random.permutation(len(X))
            folds = np.array_split(indices, k)

            mse_list = []
            loss_list = []
            best_epoch_list = []
            
            for i in range(k):
                val_idx = folds[i]
                train_idx = np.hstack([folds[j] for j in range(k) if j != i])

                X_train, y_train = X[train_idx], y[train_idx]
                X_val,   y_val   = X[val_idx], y[val_idx]

                val_loss, val_mse_epoch, best_epoch = train_model(
                    X_train, y_train,
                    X_val,   y_val,
                    learning_rate=0.001,
                    batch_size=batch_size,
                    epochs=150,
                    output_function=output_function,
                    output_func_der=output_func_der
                )
                mse_list.append(val_mse_epoch)
                loss_list.append(val_loss)
                best_epoch_list.append(best_epoch)
                
            max_epoch = max(best_epoch_list)
            avg_mse = float(np.mean(mse_list))
            avg_loss = float(np.mean(loss_list))

            print(f"batch={batch_size}, out={out_func} -> avg MSE: {avg_mse:.6f}, avg Poisson: {avg_loss:.6f}")

            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                best_batch_size = batch_size
                best_out_func = out_func
                best_epoch_final = max_epoch
        
    print("Best config:", best_batch_size, best_out_func, best_avg_loss, best_epoch_final)
    return best_batch_size, best_out_func,best_avg_loss, best_epoch_final

def predict(X, parameters, output_function):
    y_pred, _ = feed_forward(X, parameters, output_function)
    return y_pred

def train_final_model(X_train, y_train,learning_rate, batch_size, epochs,output_function, output_func_der):
    parameters = initialize_parameters(
        input_layer = X_train.shape[1],
        hidden_layer1_neurons = 28,
        hidden_layer2_neurons = 28,
        output_layer = 1
    )
    
    adam_state = init_adam_state(parameters)
    
    for epoch in range(epochs):
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for i in range(0, len(X_shuffled), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
        
            _, values = feed_forward(X_batch, parameters, output_function)

            gradients = backward_propagation(
                X_batch, y_batch, parameters, values, output_func_der
            )

            parameters, adam_state = adam_update(
                parameters, gradients, adam_state,
                learning_rate=learning_rate
            )
        
    return parameters