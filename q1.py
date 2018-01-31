from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn as sk
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):

        plt.subplot(3, 5, i + 1)
        plt.xlabel(features[i])
        plt.ylabel("Price in 1000's")
        plt.title(features[i] + ' vs. house prices')
        plt.plot([item[i] for item in X],y, 'bo')
    
    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    # Remember to use np.linalg.solve instead of inverting!
    X = np.array(X)
    y = np.array(Y)
    print(X.shape)
    print(y.shape)
    print(y)
    X_t = X.transpose()
    a = np.dot(X_t, X)
    b = np.dot(X_t, y)
    w = np.linalg.solve(a,b)

    # Using Inverse
    # Xt = X.transpose()
    # product = np.dot(Xt, X)
    # theInverse = np.linalg.inv(product)
    # w = np.dot(np.dot(theInverse, Xt), Y)

    return w

def data_split (X, y):
    #splits data into training and test
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []

    for i in range(0, 506):
        c = np.random.random_integers(1,100)
        if c <= 80 and c>=1:
            train_set_x.append(X[i])
            train_set_y.append(y[i])
        elif (c>80 and c<=100):
            test_set_x.append(X[i])
            test_set_y.append(y[i])

    return train_set_x, train_set_y, test_set_x, test_set_y

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    # Visualize the features
    visualize(X, y, features)

    # split training and test
    train_set_x, train_set_y, test_set_x, test_set_y = data_split (X, y)
    train_set_x1 = np.concatenate((np.ones((len(train_set_x),1)),train_set_x),axis=1)
    test_set_x1 = np.concatenate((np.ones((len(test_set_x),1)),test_set_x),axis=1)

    # # Fit regression model
    w = fit_regression(train_set_x1, train_set_y)
    print("Weights:")
    # w2 = fit_regression(X[:405], y[:405])
    print(w)


    # Compute fitted values, MSE, etc.
    prediction = np.dot(test_set_x1,w)

    # MSE more useful if we are concerned about large errors whose consequences are much bigger than equivalent smaller
    # ones. MSE emphasizes the extremes.
    mse = np.mean((test_set_y - prediction) ** 2)
    print("Mean Square Error is: ")
    print(mse)
    # Mean Absolute Error MAE is more robust to outliers since it does not make use of square. It gives equal weights.
    mae = mean_absolute_error(test_set_y, prediction)
    print("Mean Absolute Error is: ")
    print(mae)
    # R^2
    r2 = r2_score(test_set_y, prediction)
    print("R^2 is: ")
    print(r2)

if __name__ == "__main__":
    main()



