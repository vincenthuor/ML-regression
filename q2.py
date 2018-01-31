from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
import scipy as sp
from scipy import misc
from numpy import matlib
import random
import math

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j, tau in enumerate(taus):
        predictions = np.array([LRLS(x_test[i, :].reshape(d, 1), x_train, y_train, tau) \
                                for i in range(N_test)])
        losses[j] = ((predictions.flatten() - y_test.flatten()) ** 2).mean()
    return losses

def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    # shuffle data set
    c = list(zip(x, y))
    random.shuffle(c)
    x, y = zip(*c)

    A = []
    for X_train, y_train, X_valid, y_valid in k_fold_generator(x, y, k):
        loss_to_tau = run_on_fold(X_valid, y_valid, X_train, y_train, taus)
        A.append(loss_to_tau)

    A = np.array(A)
    average_loss = np.sum(A, axis=0)/k

    return average_loss

def k_fold_generator(X, y, k_fold):

    #iterator for each train/test interation
    subset_size = int(len(X) / k_fold)
    for k in range(k_fold):
        x_training = X[:k * subset_size] + X[(k + 1) * subset_size:]
        x_test = X[k * subset_size:][:subset_size]
        y_training = y[:k * subset_size] + y[(k + 1) * subset_size:]
        y_test = y[k * subset_size:][:subset_size]

        x_training = np.array(x_training)
        y_training = np.array(y_training)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        yield x_training, y_training, x_test, y_test

# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    test_datum = np.reshape(test_datum, (1, 14)) # reshape test_datum
    A = A_diag(test_datum, x_train, tau)

    X_t = x_train.transpose()
    X_t_A = np.dot(X_t, A)
    X_t_A_X = np.dot(X_t_A, x_train)
    lam_I = np.multiply(lam, np.matlib.identity(X_t_A_X.shape[0]))

    x_1 = np.add(X_t_A_X, lam_I)
    b = np.dot(X_t, A)
    c = np.dot(b, y_train)
    w = np.linalg.solve(x_1,c)
    # w = [4.06166562e+01   1.60915786e+00 - 1.07009701e-01 - 1.69266696e-01
    #      - 1.35737499e+00   1.40502612e+01   7.05931184e+00 - 1.12058271e-02
    #       2.33149800e-01   2.18912127e-01 - 7.38435848e-02 - 7.42063634e-01
    #     - 8.21594445e-02 - 1.09458030e-01]

    # transposed already
    y_hat = predict_y(test_datum, w)

    return y_hat

def predict_y(x_train, w):

    return np.dot(x_train, w)[0]

def A_diag(test_datum, x_train, tau):
    #returns A_ii matrix
    num = l2(test_datum, x_train) / (-2*(tau ** 2))
    denom1 = sp.misc.logsumexp(num)
    final = np.exp(np.subtract(num, denom1))

    return np.diag(final[0])

if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)

    print("______")

    losses = run_k_fold(x, y, taus, 5)
    print(losses)
    print("min loss = {}".format(losses.min()))

    plt.xlabel('Tau')
    plt.ylabel("Losses")
    plt.title('Tau vs. losses for K = 5')
    plt.plot(taus, losses)
    plt.show()