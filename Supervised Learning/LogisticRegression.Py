import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
    
    def initialize_parameters(self, X):
        self.b = 0
        self.w = np.zeros(X.shape[1])

    def sigmoid(self, z):
        s = 1 / ( 1 + np.exp(-z) )

        return s

    def compute_predictions(self, X):
        
        z = np.dot(X, self.w) + self.b
        y_hat = self.sigmoid(z)
        
        return y_hat

    def compute_cost(self, y_hat, y):
        m = len(y)
        L = (-y * np.log(y_hat)) - ((1 - y) * np.log(1 - y_hat))

        cost = (1/m) * np.sum( L )

        return cost

    def compute_gradients(self, X, y, y_hat):
        
        m = len(y)

        dw = 1/m * np.dot(X.T, (y_hat - y))
        db = 1/m * np.sum(y_hat - y)

        return dw, db

    def update_parameters(self, dw, db):
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def train(self, X, y):
        
        self.initialize_parameters(X)

        for i in range(self.iterations):
            y_hat = self.compute_predictions(X)
            cost = self.compute_cost(y_hat, y)

            dw, db = self.compute_gradients(X, y, y_hat)

            self.update_parameters(dw, db)

            if i % 100 == 0:
                print(f"Iteration {i:4d} | Cost: {cost:.6f}")


    def predict(self, X, threshold=0.5):
        z = np.dot(X, self.w) + self.b
        y_prob = self.sigmoid(z)

        y_pred = (y_prob >= threshold).astype(int)

        return y_pred


X = np.array([
    [18, 20000],
    [20, 22000],
    [22, 25000],
    [25, 27000],
    [28, 30000],
    [30, 32000],
    [35, 35000],
    [40, 40000],
    [50, 45000],
    [60, 50000]
])

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

def min_max_scale(X):
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        return (X - X_min) / (X_max - X_min)

X_scaled = min_max_scale(X)

model = LogisticRegression(learning_rate=0.1, iterations=1000)
model.train(X_scaled, y)

y_pred = model.predict(X_scaled)

print("\nPredictions:", y_pred)
print("True Labels:", y)

from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y, y_pred)
print(cm)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall   :", recall_score(y, y_pred))
print("F1 Score :", f1_score(y, y_pred))

plt.scatter(X[:, 0], y, color='blue', label='True')
plt.scatter(X[:, 0], y_pred, color='red', marker='x', label='Predicted')
plt.xlabel("Age")
plt.ylabel("Buy (0 or 1)")
plt.title("Logistic Regression Predictions")
plt.legend()
plt.grid(True)
plt.show()



