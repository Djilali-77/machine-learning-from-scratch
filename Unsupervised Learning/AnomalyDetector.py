import numpy as np
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self, ):
        self.mean = None
        self.var = None
        self.epsilon = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)

    def gaussian_pdf(self, X):
        p = (1 / np.sqrt( 2 * np.pi * self.var )) * np.exp(- (X - self.mean)**2 / (2 * self.var))
        return np.prod(p, axis=1)

    def predict_proba(self, X):
        return self.gaussian_pdf(X)


    def select_threshold(self, y_val, p_val):
        best_epsilon = 0
        best_f1 = 0

        epsilons = np.linspace(min(p_val), max(p_val), 1000)

        for eps in epsilons:
            predictions = (p_val < eps).astype(int)

            tp = np.sum((predictions == 1) & (y_val == 1))
            fp = np.sum((predictions == 1) & (y_val == 0))
            fn = np.sum((predictions == 0) & (y_val == 1))

            if tp + fp == 0 or tp + fn == 0:
                continue

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            f1 = 2 * precision * recall / (precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                best_epsilon = eps

        self.epsilon = best_epsilon
        return best_epsilon, best_f1


    def predict(self, X):
        p = self.predict_proba(X)
        return (p < self.epsilon).astype(int)



from sklearn.model_selection import train_test_split

X_normal = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 300)

X_anomalies = np.array([[5, 5], [-6, -4], [6, -5]])

X = np.vstack((X_normal, X_anomalies))
y = np.hstack((np.zeros(len(X_normal)), np.ones(len(X_anomalies))))  # 0 = normal, 1 = anomaly

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

model = AnomalyDetector()
model.fit(X_train)

p_val = model.predict_proba(X_val)
epsilon, f1 = model.select_threshold(y_val, p_val)

print("Best epsilon:", epsilon)
print("Best F1 score:", f1)


y_pred = model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="coolwarm", s=40)
plt.title("Anomaly Detection Results")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()
