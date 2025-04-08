import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def preprocess_data():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # Save the datasets as .npy files
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)

if __name__ == "__main__":
    preprocess_data()