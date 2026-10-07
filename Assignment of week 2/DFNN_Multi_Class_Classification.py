
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 1. LOAD DATASET
covertype = fetch_ucirepo(id=31)
X = covertype.data.features
y = covertype.data.targets
X = X.values
y = y.values.ravel()
print("\nDataset Shape:", X.shape)
print("Target Shape:", y.shape)
print("Classes:", np.unique(y))

# 2. CLASS DISTRIBUTION
class_counts = pd.Series(y).value_counts().sort_index()
print("\nClass Distribution:\n", class_counts)
plt.figure(figsize=(8,5))
plt.bar(class_counts.index, class_counts.values)
plt.xlabel("Forest Cover Type")
plt.ylabel("Number of Samples")
plt.title("Class Distribution")
plt.show()
imbalance_ratio = class_counts.max() / class_counts.min()
print("Imbalance Ratio:", imbalance_ratio)


# 3. TARGET PREPARATION
y = y - 1
num_classes = 7
Y = np.zeros((len(y), num_classes))
Y[np.arange(len(y)), y] = 1
print("One-Hot Target Shape:", Y.shape)

# 4. TRAIN / VALIDATION / TEST SPLIT
X_train, X_temp, Y_train, Y_temp, y_train, y_temp = train_test_split(
    X, Y, y, test_size=0.20, random_state=42, stratify=y
)
X_val, X_test, Y_val, Y_test, y_val, y_test = train_test_split(
    X_temp, Y_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
print("\nTraining:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# 5. STANDARDIZATION
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 6. CLASS WEIGHTS
N = len(y_train)
K = num_classes
class_counts_train = np.bincount(y_train, minlength=K)
class_weights = N / (K * class_counts_train)
print("\nClass Weights:")
for i in range(K):
    print(f"Class {i}: Count={class_counts_train[i]}, Weight={class_weights[i]:.4f}")

# 7. ACTIVATION FUNCTIONS
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# 8. NETWORK ARCHITECTURE
input_size, hidden1_size, hidden2_size, hidden3_size, output_size = 54, 64, 32, 16, 7

print("\nNetwork Architecture: 54 -> 64 -> 32 -> 16 -> 7")

# 9. HE INITIALIZATION
np.random.seed(42)
W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2/input_size)
b1 = np.zeros((1, hidden1_size))

W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2/hidden1_size)
b2 = np.zeros((1, hidden2_size))

W3 = np.random.randn(hidden2_size, hidden3_size) * np.sqrt(2/hidden2_size)
b3 = np.zeros((1, hidden3_size))

W4 = np.random.randn(hidden3_size, output_size) * np.sqrt(2/hidden3_size)
b4 = np.zeros((1, output_size))

# 10. FORWARD PROPAGATION
def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = relu(Z3)
    Z4 = np.dot(A3, W4) + b4
    Y_hat = softmax(Z4)
    
    cache = {"Z1":Z1, "A1":A1, "Z2":Z2, "A2":A2, "Z3":Z3, "A3":A3, "Z4":Z4, "Y_hat":Y_hat}
    return Y_hat, cache

# 11. WEIGHTED CATEGORICAL CROSS-ENTROPY
def compute_loss(Y_true, Y_pred, labels):
    epsilon = 1e-12
    Y_pred = np.clip(Y_pred, epsilon, 1-epsilon)
    sample_weights = class_weights[labels]
    loss = -np.sum(Y_true * np.log(Y_pred), axis=1)
    return np.mean(sample_weights * loss)

# 12. BACKPROPAGATION
def backward_propagation(X, Y_true, Y_pred, cache, labels):
    global W1, W2, W3, W4
    
    Z1, A1 = cache["Z1"], cache["A1"]
    Z2, A2 = cache["Z2"], cache["A2"]
    Z3, A3 = cache["Z3"], cache["A3"]
    
    m = X.shape[0]
    sample_weights = class_weights[labels]
    
    # Output layer
    dZ4 = (Y_pred - Y_true) * sample_weights.reshape(-1,1) / m
    dW4 = np.dot(A3.T, dZ4)
    db4 = np.sum(dZ4, axis=0, keepdims=True)
    
    # Hidden Layer 3
    dA3 = np.dot(dZ4, W4.T)
    dZ3 = dA3 * relu_derivative(Z3)
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)
    
    # Hidden Layer 2
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # Hidden Layer 1
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return {"dW1":dW1, "db1":db1, "dW2":dW2, "db2":db2,
            "dW3":dW3, "db3":db3, "dW4":dW4, "db4":db4}

# 13. UPDATE PARAMETERS
def update_parameters(gradients, learning_rate):
    global W1, b1, W2, b2, W3, b3, W4, b4
    
    W1 -= learning_rate * gradients["dW1"]
    b1 -= learning_rate * gradients["db1"]
    W2 -= learning_rate * gradients["dW2"]
    b2 -= learning_rate * gradients["db2"]
    W3 -= learning_rate * gradients["dW3"]
    b3 -= learning_rate * gradients["db3"]
    W4 -= learning_rate * gradients["dW4"]
    b4 -= learning_rate * gradients["db4"]

# 14. PREDICTION
def predict(X):
    probabilities, _ = forward_propagation(X)
    predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities

# 15. TRAINING PARAMETERS
learning_rate, batch_size, epochs, patience = 0.01, 128, 100, 15
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_loss = np.inf
patience_counter = 0
best_parameters = None

# 16. TRAINING
print("\nStarting Training...\n")

for epoch in range(epochs):
    
    permutation = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[permutation]
    Y_train_shuffled = Y_train[permutation]
    y_train_shuffled = y_train[permutation]
    
    for start in range(0, len(X_train), batch_size):
        
        end = start + batch_size
        
        X_batch = X_train_shuffled[start:end]
        Y_batch = Y_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]
        
        Y_pred_batch, cache = forward_propagation(X_batch)
        
        gradients = backward_propagation(
            X_batch, Y_batch, Y_pred_batch, cache, y_batch
        )
        
        update_parameters(gradients, learning_rate)
    
    # Training performance
    train_pred, train_prob = predict(X_train)
    train_loss = compute_loss(Y_train, train_prob, y_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    # Validation performance
    val_pred, val_prob = predict(X_val)
    val_loss = compute_loss(Y_val, val_prob, y_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    
    print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        
        best_parameters = {
            "W1":W1.copy(), "b1":b1.copy(),
            "W2":W2.copy(), "b2":b2.copy(),
            "W3":W3.copy(), "b3":b3.copy(),
            "W4":W4.copy(), "b4":b4.copy()
        }
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("\nEarly stopping triggered.")
        break

# 17. RESTORE BEST PARAMETERS
W1, b1 = best_parameters["W1"], best_parameters["b1"]
W2, b2 = best_parameters["W2"], best_parameters["b2"]
W3, b3 = best_parameters["W3"], best_parameters["b3"]
W4, b4 = best_parameters["W4"], best_parameters["b4"]

# 18. LOSS GRAPH
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.show()

# 19. ACCURACY GRAPH
plt.figure(figsize=(8,5))
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid()
plt.show()

# 20. TEST PREDICTION
y_pred, y_prob = predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("\n" + "="*60)
print("FINAL TEST RESULTS")
print("="*60)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")

# 21. CLASSIFICATION REPORT
class_names = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine",
               "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]

print("\nClassification Report:\n")
print(classification_report(
    y_test, y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
))

# 22. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 23. PER-CLASS METRICS
precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

metrics_df = pd.DataFrame({
    "Class": class_names,
    "Precision": precision_per_class,
    "Recall": recall_per_class,
    "F1 Score": f1_per_class
})

print("\nPer-Class Metrics:\n")
print(metrics_df.to_string(index=False))

# 24. F1 SCORE GRAPH
plt.figure(figsize=(10,5))
plt.bar(class_names, f1_per_class)
plt.xlabel("Forest Cover Type")
plt.ylabel("F1 Score")
plt.title("Per-Class F1 Score")
plt.ylim(0,1)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 25. PRECISION VS RECALL
x_axis = np.arange(len(class_names))
width = 0.35

plt.figure(figsize=(10,5))
plt.bar(x_axis-width/2, precision_per_class, width, label="Precision")
plt.bar(x_axis+width/2, recall_per_class, width, label="Recall")
plt.xlabel("Forest Cover Type")
plt.ylabel("Score")
plt.title("Precision vs Recall")
plt.xticks(x_axis, class_names, rotation=45, ha="right")
plt.ylim(0,1)
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 26. ACTUAL VS PREDICTED CLASS DISTRIBUTION
true_counts = np.bincount(y_test, minlength=7)
pred_counts = np.bincount(y_pred, minlength=7)

x_axis = np.arange(7)

plt.figure(figsize=(10,5))
plt.bar(x_axis-width/2, true_counts, width, label="Actual")
plt.bar(x_axis+width/2, pred_counts, width, label="Predicted")
plt.xlabel("Forest Cover Type")
plt.ylabel("Number of Samples")
plt.title("Actual vs Predicted Class Distribution")
plt.xticks(x_axis, class_names, rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

# 27. FINAL SUMMARY
print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
print("Architecture: 54 -> 64 -> 32 -> 16 -> 7")
print("Hidden Activation: ReLU")
print("Output Activation: Softmax")
print("Loss: Weighted Categorical Cross-Entropy")
print("Optimizer: Mini-Batch SGD")
print("Batch Size:", batch_size)
print("Learning Rate:", learning_rate)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Macro F1 Score: {macro_f1:.4f}")
print("\nDFNN Multi-Class Classification completed.")