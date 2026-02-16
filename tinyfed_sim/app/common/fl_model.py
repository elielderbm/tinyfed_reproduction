import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s*(1.0 - s)

class MLPBinary4_16_8_4_2:
    def __init__(self, input_dim=4, h1=16, h2=8, h3=4, out_dim=2, lr=0.05, seed=42):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W1 = rng.normal(0, np.sqrt(2/(input_dim+h1)), size=(input_dim, h1))
        self.b1 = np.zeros((1, h1))
        self.W2 = rng.normal(0, np.sqrt(2/(h1+h2)), size=(h1, h2))
        self.b2 = np.zeros((1, h2))
        self.W3 = rng.normal(0, np.sqrt(2/(h2+h3)), size=(h2, h3))
        self.b3 = np.zeros((1, h3))
        self.W4 = rng.normal(0, np.sqrt(2/(h3+out_dim)), size=(h3, out_dim))
        self.b4 = np.zeros((1, out_dim))

    def get_weights(self):
        return {k: v.tolist() for k, v in {
            "W1":self.W1, "b1":self.b1, "W2":self.W2, "b2":self.b2,
            "W3":self.W3, "b3":self.b3, "W4":self.W4, "b4":self.b4
        }.items()}

    def set_weights(self, weights):
        self.W1 = np.array(weights["W1"], dtype=float)
        self.b1 = np.array(weights["b1"], dtype=float)
        self.W2 = np.array(weights["W2"], dtype=float)
        self.b2 = np.array(weights["b2"], dtype=float)
        self.W3 = np.array(weights["W3"], dtype=float)
        self.b3 = np.array(weights["b3"], dtype=float)
        self.W4 = np.array(weights["W4"], dtype=float)
        self.b4 = np.array(weights["b4"], dtype=float)

    @staticmethod
    def _softmax(z):
        z_stable = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / (exp_z.sum(axis=1, keepdims=True) + 1e-12)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1; a1 = sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2; a2 = sigmoid(z2)
        z3 = a2 @ self.W3 + self.b3; a3 = sigmoid(z3)
        z4 = a3 @ self.W4 + self.b4; a4 = self._softmax(z4)
        cache = (X, z1,a1, z2,a2, z3,a3, z4,a4)
        return a4, cache

    def loss_and_grads(self, X, y_onehot):
        yhat, cache = self.forward(X)

        # ---- Ajuste: peso maior para anomalias ----
        weights = np.where(y_onehot[:,1] == 1, 3.0, 1.0)  # anomalia pesa 3x
        loss = -np.mean(weights * np.sum(y_onehot * np.log(yhat + 1e-12), axis=1))

        (X, z1,a1, z2,a2, z3,a3, z4,a4) = cache
        dz4 = (a4 - y_onehot) / X.shape[0]

        dW4 = (a3.T @ dz4)
        db4 = dz4.sum(axis=0, keepdims=True)

        da3 = dz4 @ self.W4.T
        dz3 = da3 * dsigmoid(z3)
        dW3 = a2.T @ dz3
        db3 = dz3.sum(axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * dsigmoid(z2)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * dsigmoid(z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        grads = (dW1,db1,dW2,db2,dW3,db3,dW4,db4)
        return loss, grads, yhat

    def step(self, grads):
        (dW1,db1,dW2,db2,dW3,db3,dW4,db4) = grads
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4

    @staticmethod
    def one_hot(y, num_classes=2):
        oh = np.zeros((y.shape[0], num_classes), dtype=float)
        oh[np.arange(y.shape[0]), y.reshape(-1)] = 1.0
        return oh

    def predict(self, X):
        yhat, _ = self.forward(X)
        return np.argmax(yhat, axis=1), yhat[:,1]
