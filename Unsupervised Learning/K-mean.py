import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iterations = 1000):
        self.k = k
        self.max_iterations = max_iterations
        self.c = None

    def initialize_centroids(self, X):
        
        i = np.random.choice(X.shape[0], self.k, replace=False)
        self.c = X[i]

        plt.scatter(self.c[:, 0], self.c[:, 1], c='black', s=200, alpha=0.6, marker='X', label='Centroids random')
 
    def assign_clusters(self, X):
        dist = []

        for x in X:
            d = [np.sum ((x - cent) ** 2) for cent in self.c]
            min_d = np.argmin(d)

            dist.append(min_d)

        return np.array(dist)

    def update_centroids(self, X, dist):
        new_cent = []

        for i in range(self.k):

            cluster_points = X[dist == i]


            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)
            else:
                new_centroid = X[np.random.randint(0, X.shape[0])]

            new_cent.append(new_centroid)

        self.c = np.array(new_cent)

    def fit(self, X):
        
        self.initialize_centroids(X)

        for i in range(self.max_iterations):

            dist = self.assign_clusters(X)
            self.update_centroids(X, dist)

    def predict(self, X):
        return self.assign_clusters(X)





from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

model = KMeans(k=3, max_iterations=100)
model.fit(X)
dist = model.predict(X)


plt.scatter(X[:, 0], X[:, 1], c=dist, cmap='viridis', s=30)

plt.scatter(model.c[:, 0], model.c[:, 1], c='red', s=200, alpha=0.6, marker='X', label='Centroids')

plt.title("K-Means Clustering Result")
plt.legend()
plt.grid(True)
plt.show()


print("Final Centroids:")
print(model.c)








