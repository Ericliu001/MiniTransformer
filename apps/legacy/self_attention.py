import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy_loss(pred, target):
    return -np.sum(target * np.log(pred + 1e-9))  # Avoid log(0)

# Hyperparameters
learning_rate = 0.1
epochs = 100

# Input: 3 words, each 4-dimensional embedding
X = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 1, 1, 1]])

# Initialize weight matrices randomly
np.random.seed(42)
W_Q = np.random.randn(4, 3)
W_K = np.random.randn(4, 3)
W_V = np.random.randn(4, 3)

# Target attention matrix (supervised example)
target_attention = np.array([[1, 0, 0],  # Word 1 attends to itself
                             [0, 1, 0],  # Word 2 attends to itself
                             [0, 0, 1]]) # Word 3 attends to itself

# Training loop
for epoch in range(epochs):
    # Forward pass
    Q = np.dot(X, W_Q)
    K = np.dot(X, W_K)
    V = np.dot(X, W_V)

    # Compute scaled dot-product attention
    S = np.dot(Q, K.T) / np.sqrt(3)
    A = softmax(S)  # Apply softmax
    Z = np.dot(A, V)  # Compute output

    # Compute loss
    loss = cross_entropy_loss(A, target_attention)

    # Compute gradients
    grad_A = A - target_attention  # Gradient of loss w.r.t. attention matrix
    grad_S = grad_A * A * (1 - A)  # Gradient of softmax
    grad_Q = np.dot(grad_S, K) / np.sqrt(3)  # Gradient of Q
    grad_K = np.dot(grad_S.T, Q) / np.sqrt(3)  # Gradient of K
    grad_V = np.dot(A.T, grad_A)  # Gradient of V

    # Update weights using gradient descent
    W_Q -= learning_rate * np.dot(X.T, grad_Q)
    W_K -= learning_rate * np.dot(X.T, grad_K)
    W_V -= learning_rate * np.dot(X.T, grad_V)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


# Print the trained self-attention matrix
print("Trained Self-Attention Matrix:")
print(np.round(A, 3))  # Print values rounded to 3 decimal places


# Final learned self-attention matrix
# import matplotlib.pyplot as plt
# plt.figure(figsize=(6, 5))
# plt.imshow(A, cmap='Blues', aspect='auto')
# plt.colorbar(label='Attention Weight')
# plt.xticks(ticks=[0, 1, 2], labels=["Word 1", "Word 2", "Word 3"])
# plt.yticks(ticks=[0, 1, 2], labels=["Word 1", "Word 2", "Word 3"])
# plt.xlabel("Attention To")
# plt.ylabel("Attention From")
# plt.title("Trained Self-Attention Matrix")
# plt.show()
