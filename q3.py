import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCHES = 50

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, X, y, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch    


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    X = np.array(X)
    y = np.array(y)
    w = np.array(w)

    X_t = X.transpose()
    a = np.dot(np.dot(X_t, X), w)
    b = np.dot(X_t, y)

    return np.multiply(2/len(X), a-b)

def batch_reg(X, y, w, batch_size):

    #returns the w for a specific batch size
    X = np.array(X)
    y = np.array(y)
    w = np.array(w)

    X_t = X.transpose()
    a = X_t.dot(X).dot(w)
    b = X.transpose().dot(y)


    # return (a - b)*(2/BATCHES)
    return (a - b).dot((2/batch_size))


def true_gradient(X, y, w):
    '''
    True gradient over all N=506
    '''
    X = np.array(X)
    y = np.array(y)
    w = np.array(w)

    X_t = X.transpose()
    a = X_t.dot(X).dot(w)
    b = X.transpose().dot(y)

    # return (a - b)*(2/BATCHES)
    return (a - b).dot((2/506))

def mini_sgd_gradient_var(X, y, w, batch_sampler):
    #calculates the variance of a minibatch
    # returns vector of variance of size 500
    var_temp = []
    for i in range(1, 401):
        K = 500
        list_w = []

        for j in range(K):
            X_b, y_b = batch_sampler.get_batch(X, y, i)
            batch_grad = lin_reg_gradient(X_b, y_b, w)
            list_w.append(batch_grad[3]) # appends one w to vector of 500
        mean = np.mean(list_w) # mean of the 500

        list_w = np.array(list_w)
        var = np.sum(((list_w - mean)**2), axis=0) / K
        var_temp.append(var)

    return var_temp

def mini_batch(X, y, w, batch_sampler, m):
    #returns the expected value of the minibatch_gradient
    # mini-SGD
    K = 500
    SUM = np.zeros(13)
    for i in range(500):
        X_b, y_b = batch_sampler.get_batch(X, y, m)
        batch_grad = lin_reg_gradient(X_b, y_b, w)
        # print(batch_grad)
        SUM += batch_grad
    mean = SUM / K
    return mean

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()

    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    # True gradient
    true_grad = true_gradient(X, y, w)
    wj = true_grad[0]
    print('True Gradient:')
    print(wj)

    # mini-batch grad
    Expected_value_mini_batch = mini_batch(X, y, w, batch_sampler, 50)
    print('Expected_value_mini_batchs Gradient:')
    print(Expected_value_mini_batch[0])
    cosine_sim = cosine_similarity(Expected_value_mini_batch, true_gradient(X, y, w))
    print('Cosine similarity:')
    print(cosine_sim)
    mse = np.mean((Expected_value_mini_batch - true_gradient(X, y, w)) ** 2)
    print("Mean Square Error is: ")
    print(mse)

    # vector of variance for 500times sampled, of varying m from 1-400
    b = mini_sgd_gradient_var(X, y, w, batch_sampler)
    inc = range(2, 401)
    print('Variance vector: ')
    print(b)

    plt.xlabel('log(m)')
    plt.ylabel("Variance")
    plt.title('Sample Variance vs Batch Size')
    plt.plot(np.log(inc), np.log(b[1:]))
    plt.show()

if __name__ == '__main__':
    main()
