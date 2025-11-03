import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, lambdaa = 0.001, regularization = None):

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        self.lambdaa = lambdaa
        self.regularization = regularization

    def initialize_parameters(self, X):
        self.b = 0
        self.w = np.zeros(X.shape[1])

    def Min_Max_Scaling(self, a):

        a_min = np.min(a, axis=0)
        a_max = np.max(a, axis=0)

        a_scaled = (a - a_min) / (a_max - a_min)
        
        return a_scaled
    
    def StandardScaler(self, a):
        a_mean = np.mean(a, axis=0)
        s = np.std(a, axis=0)

        a_scaled = (a - a_mean) / s

        return a_scaled
    
    def regularization_l2(self, y, y_hat, w):

        m = len(y)
        lambdaa = self.lambdaa
        J = (1 / (2 * m)) * np.sum((y_hat - y) ** 2) + lambdaa/(2*m) * np.sum(w**2)
        return J
    
    def regularization_l1(self, y, y_hat, w):

        m = len(y)
        lambdaa = self.lambdaa
        J = (1 / (2 * m)) * np.sum((y_hat - y) ** 2) + lambdaa/(2*m) * np.sum(np.abs(w))
        return J
    
    @staticmethod
    def polynomial_features(X, degree):
        X_poly = X.copy()
        for d in range(2, degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly


    def compute_predictions(self, X):
        
        y_hat = np.dot(X, self.w) + self.b

        return y_hat

    def compute_cost(self, y_hat, y):
        m = len(y)
        if self.regularization == 'l2':
            J = self.regularization_l2(y, y_hat, self.w)
        elif self.regularization == 'l1':
            J = self.regularization_l1(y, y_hat, self.w)
        else:
            J = (1 / (2 * m)) * np.sum((y_hat - y) ** 2)
        return J

    def compute_gradients(self, X, y, y_hat):
        m = len(y)

        if self.regularization == 'l2':
            dw = (1/m) * np.dot(X.T, (y_hat - y)) + (self.lambdaa / m) * self.w
        elif self.regularization == 'l1':
            dw = (1/m) * np.dot(X.T, (y_hat - y)) + (self.lambdaa / m) * np.sign(self.w)
        else:
            dw = 1/m * np.dot(X.T, (y_hat - y))

        db = 1/m * np.sum((y_hat - y))

        return dw, db

    def update_parameters(self, dw, db):

        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    
    def train(self, X, y):

        self.initialize_parameters(X)

        for i in range(self.iterations):

            y_hat = self.compute_predictions(X)
            cost = self.compute_cost(y_hat, y)

            if i % 100 == 0:
                print(f"Iteration {i:4d} | Cost: {cost:.6f}")
            
            dw, db = self.compute_gradients(X, y, y_hat)

            self.update_parameters(dw, db)
        

    def predict(self, X):
        y_pred = np.dot(X, self.w) + self.b

        return y_pred

while True:
    c = int(input("Choose an option:\n1. Linear Regression\n2. Polynomial Regression\nEnter 1 or 2: "))
    try : 
        if c==1 : 

            #Linear regression
            X = np.array([
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5]
            ])

            y = np.array( [2, 4, 6, 8] )

            model = LinearRegression(learning_rate=0.01, iterations=1000, lambdaa=0.01, regularization='l2')

            X_scaled = model.Min_Max_Scaling(X)

            model.train(X_scaled, y)

            y_pred = model.predict(X_scaled)

            print("Learned weights:", model.w)
            print("Learned bias:", model.b)

            plt.scatter(X_scaled[:, 0], y, color='blue', label='True data')
            plt.plot(X_scaled[:, 0], y_pred, color='red', label='Predicted line')

            plt.legend()
            plt.show()

            break
        elif c == 2:

            #Ploymial regression

            np.random.seed(0)
            A = np.linspace(-3, 3, 100).reshape(-1, 1)
            b = 0.5 * A**3 - A**2 + A + np.random.randn(100, 1) * 2
            b = b.reshape(-1)

            model2 = LinearRegression(learning_rate=0.01, iterations=1000, lambdaa=0.01, regularization='l2')


            A_poly = model2.polynomial_features(A, degree=3)
            A_scaled = model2.Min_Max_Scaling(A_poly)

            model2.train(A_scaled, b)

            b_pred = model2.predict(A_scaled)

            print("Learned weights:", model2.w)
            print("Learned bias:", model2.b)


            plt.scatter(A, b, color='blue', label='True data')
            plt.plot(A, b_pred, color='red', label='Predicted curve')
            plt.legend()
            plt.title("Polynomial Regression Fit")
            plt.show()
            break

        else : 
            print("Choose 1 or 2")
    except ValueError:
        print("Invalid input! Please enter a number (1 or 2).")

