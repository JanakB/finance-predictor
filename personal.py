import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PersonalFinanceManager:
    def __init__(self):
        self.weights = None

    def train_linear_regression(self, X, y):
        # Step 5: Model Training
        X_transpose = np.transpose(X)
        X_transpose_X_inv = np.linalg.inv(np.dot(X_transpose, X))
        self.weights = np.dot(np.dot(X_transpose_X_inv, X_transpose), y)

    def evaluate_model(self, X, y):
        # Step 6: Model Evaluation
        y_pred = np.dot(X, self.weights)
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse

    def predict_expenses(self, new_features):
        # Step 7: Prediction
        predicted_expenses = np.dot(new_features, self.weights)
        return predicted_expenses

    def visualize_results(self, X, y):
        # Step 8: Visualization
        plt.scatter(y, np.dot(X, self.weights))
        plt.xlabel("Actual Expenses")
        plt.ylabel("Predicted Expenses")
        plt.title("Actual vs Predicted Expenses")
        plt.show()

def main():
    # Step 1: Data Collection
    # Load your CSV data
    df = pd.read_csv("Budget.csv")

    # Assuming the columns in your CSV are "Income", "Previous Expenses", "Time of Year", and "Expenses"
    income = df["Income"].values
    prev_expenses = df["Previous Expenses"].values
    time_of_year = df["Time of Year"].values
    expenses = df["Expenses"].values

    # Step 3: Feature Selection
    X = np.column_stack((income, prev_expenses, time_of_year))

    # Instantiate the PersonalFinanceManager
    manager = PersonalFinanceManager()

    # Step 4: Split Data
    # For simplicity, let's not split the data as we have a small dataset

    # Train the model
    manager.train_linear_regression(X, expenses)

    # Evaluate the model
    train_rmse = manager.evaluate_model(X, expenses)
    print("Root Mean Squared Error on Training Data:", train_rmse)

    # Step 7: Prediction
    # Let's predict expenses for a new month
    new_income = 65000
    new_prev_expenses = 23000
    new_time_of_year = 6  # Assuming 6 for June
    new_features = np.array([new_income, new_prev_expenses, new_time_of_year])
    predicted_expenses = manager.predict_expenses(new_features)
    print("Predicted expenses for the new month:", predicted_expenses)

    # Step 8: Visualization
    manager.visualize_results(X, expenses)

if __name__ == "__main__":
    main() 
