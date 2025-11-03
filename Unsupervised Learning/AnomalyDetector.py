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

        # Try 1000 evenly spaced values between min and max of predicted probabilities
        epsilons = np.linspace(min(p_val), max(p_val), 1000)

        for eps in epsilons:
            # Step 1: Predict anomaly (1) if probability < epsilon
            predictions = (p_val < eps).astype(int)

            # Step 2: Count TP, FP, FN
            tp = np.sum((predictions == 1) & (y_val == 1))
            fp = np.sum((predictions == 1) & (y_val == 0))
            fn = np.sum((predictions == 0) & (y_val == 1))

            # Step 3: Compute precision and recall (avoid division by zero)
            if tp + fp == 0 or tp + fn == 0:
                continue

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            # Step 4: Compute F1 score
            f1 = 2 * precision * recall / (precision + recall)

            # Step 5: Keep best F1 and epsilon
            if f1 > best_f1:
                best_f1 = f1
                best_epsilon = eps

        # Save the best epsilon
        self.epsilon = best_epsilon
        return best_epsilon, best_f1


    def predict(self, X):
        p = self.predict_proba(X)
        return (p < self.epsilon).astype(int)



from sklearn.model_selection import train_test_split

# Generate normal data
X_normal = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 300)

# Add a few anomalies
X_anomalies = np.array([[5, 5], [-6, -4], [6, -5]])

# Combine data
X = np.vstack((X_normal, X_anomalies))
y = np.hstack((np.zeros(len(X_normal)), np.ones(len(X_anomalies))))  # 0 = normal, 1 = anomaly

# Split into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = AnomalyDetector()
model.fit(X_train)

# Evaluate on validation set
p_val = model.predict_proba(X_val)
epsilon, f1 = model.select_threshold(y_val, p_val)

print("Best epsilon:", epsilon)
print("Best F1 score:", f1)

# Predict anomalies on full data
y_pred = model.predict(X)

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="coolwarm", s=40)
plt.title("Anomaly Detection Results")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()
