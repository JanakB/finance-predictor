import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=2000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.theta = [0, 0]
        self.X = X.reshape(-1, 1)
        self.y = y.reshape(-1, 1)

        for i in range(self.iterations):
            z = np.dot(self.X, self.theta)
            self.theta[0] = self.theta[0] - self.learning_rate * np.sum((z - self.y) * self.X) / len(X)
            self.theta[1] = self.theta[1] - self.learning_rate * np.sum(z - self.y) / len(X)

    def predict(self, X):
        return np.dot(X, self.theta)

    def get_params(self):
        return self.theta

if __name__ == "__main__":
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 5, 4, 5])

    model = LinearRegression()
    model.fit(X, y)

    print("Weights for the respective features are :", model.get_params())