data = pd.read_csv('wine.csv')
data = data[1:]
data.head()

data = np.array(data)
m, n = data.shape
print(n, m)

# Byter så att varje sample blir en kolumn istället för rad. Skall tydligen vara enklare att jobba med då?
k = len(data)
print(k)

data_dev = data[0:100].T # .T är transpose som ändrar rader till kolumner 
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[100:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

print(X_train[:,0].shape) # Första kolumnen 

def init_params():
    # Första parametern är antal neuroner i lager 1
    W1 = np.random.randn(10, 13)
    b1 = np.random.randn(10, 1)
    W2 = np.random.randn(10, 10)
    b2 = np.random.randn(10, 1)
    return W1, b1, W2, b2

def softmax(Z): 
    return np.exp(Z) / np.sum(np.exp(Z))

def forward_prop(W1, b1, W2, b2, X): # Whats X?
    Z1 = W1.dot(X) + b1
    A1 = activate(Z1, selected_function = "relu")
    Z2 = W2.dot(A1) + b2
    A2 = softmax(A1)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros(Y.size, Y.max() + 1)
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, 2)
    dZ1 = dW2.T.dot(dZ2) * d_activate(Z1, selected_function = "relu")
    dW1 = 1 / m * dZ2.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, 2)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(prediction, Y):
    return np.sum(prediction == Y) / Y.size

def gradient_decent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0: 
            print(f"Iteration: {i}")
            print("Accuracy:", get_accuracy(get_predictions(A2), Y))
    
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_decent(X_train, Y_train, 500, 0.1)




















# Load dataset, skipping the invalid row
data = pd.read_csv('wine.csv', skiprows=[1])  # Skip the row with "r,r,r,..."
data = data.astype(float)  # Ensure all columns are numeric
data['class'] = data['class'].astype(int)  # Convert labels to integers

# Convert to NumPy array
data = data.to_numpy()
m, n = data.shape  # Get dimensions

# Separate features and labels before transposing
Y_dev = data[0:100, -1].astype(int)  # Last column as int
X_dev = data[0:100, :-1].T  # Everything except last column, then transpose

Y_train = data[100:m, -1].astype(int)
X_train = data[100:m, :-1].T

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")  # Debug

print(X_train[:,0].shape) # Första kolumnen 

def init_params():
    # Första parametern är antal neuroner i lager 1
    W1 = np.random.randn(10, 13)
    b1 = np.random.randn(10, 1)
    W2 = np.random.randn(10, 10)
    b2 = np.random.randn(10, 1)
    return W1, b1, W2, b2

def activate(activations, selected_function = "none"):
    y = activations
    if selected_function == "none":
        y = activations
    elif selected_function == "relu":
        y = np.maximum(0, activations)
    elif selected_function == "elu":
        alpha = 1
        y = np.where(activations > 0, activations, alpha * (np.exp(activations) - 1))
    return y

def d_activate(activations, selected_function = "none"):
    dy = 0
    if selected_function == "none":
        dy = np.ones_like(activations)
    elif selected_function == "relu":
        dy =  np.where(activations > 0, 1, 0)
    elif selected_function == "elu" :
        alpha = 1
        dy = np.where(activations > 0, 1, alpha * np.exp(activations))
    return dy

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = activate(Z1, selected_function="relu")
    Z2 = W2.dot(A1) + b2
    A2 = softmax(A1)  # ❌ WRONG: Should be softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y, num_classes):
    Y -= Y.min()  # Shift labels if necessary

    Y = Y.astype(int)  # Convert Y to integer class labels
    one_hot_Y = np.zeros((num_classes, Y.size))  # (num_classes, num_samples)
    one_hot_Y[Y, np.arange(Y.size)] = 1  # Assign 1s
    return one_hot_Y

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    num_classes = A2.shape[0]
    one_hot_Y = one_hot(Y, num_classes)
    m = X.shape[1]

    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = W2.T.dot(dZ2)
    dZ1 = dA1 * d_activate(Z1, selected_function="relu")
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(prediction, Y):
    return np.sum(prediction == Y) / Y.size

def gradient_decent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0: 
            print(f"Iteration: {i}")
            print("Accuracy:", get_accuracy(get_predictions(A2), Y))
    
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_decent(X_train, Y_train, 500, 0.1)