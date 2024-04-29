import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None

    def fit(self, X, y):
        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Initialize theta
        self.theta = np.zeros(X.shape[1])
        
        # Gradient Descent
        for _ in range(self.iterations):
            gradient = np.dot(X.T, np.dot(X, self.theta) - y) / y.size
            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]
        
        return np.dot(X, self.theta)

# Load the dataset
df = pd.read_csv("Budget.csv")

# Extract features and target
X = df["Category"].values.reshape(-1, 1)  # Assuming "Category" is the name of the category column
y = df["Budget"].values  # Assuming "Budget" is the name of the budget column

# Convert category to numerical values
categories = df["Category"].unique()
category_mapping = {category: i for i, category in enumerate(categories)}
X = np.array([category_mapping[category] for category in X.flatten()]).reshape(-1, 1)
categories = df["Category"].unique()
print("Unique categories:", categories)


# Instantiate and train the model
model = LinearRegression()
model.fit(X, y)

# Predictions
new_categories = np.array(['F', 'G'])  # Example new categories
new_X = np.array([category_mapping[category] for category in new_categories]).reshape(-1, 1)
predictions = model.predict(new_X)
print("Predicted budgets for new categories:", predictions)
