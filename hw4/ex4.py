import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

training_set = pd.read_csv('training_set.csv')
test_set = pd.read_csv('test_set.csv')
X_training, y_training = training_set[['x1', 'x2']].values, training_set['y'].values
X_test, y_test = test_set[['x1', 'x2']].values, test_set['y'].values


def compute_cost(X, y, theta):
    J = 0
    left_side = y * np.inner(X, theta)
    right_side = np.log(1 + (np.e ** np.inner(X, theta)))
    J = right_side - left_side
    return np.sum(J) / X.shape[0]


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, theta_shape, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.theta = np.random.random(size=theta_shape)
        self.cost_list = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Updating the theta vector in each iteration using gradient descent.
        Store the theta vector in an attribute of the LogisticRegressionGD object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        J_history = []
        theta_copy = self.theta.copy()
        k = 0
        curr_cost = compute_cost(X, y, theta_copy)
        J_history.append(0)
        while (abs(curr_cost - J_history[len(J_history) - 1]) > self.eps) and (k < self.n_iter):
            if k == 0:
                J_history.pop()
            k += 1
            J_history.append(curr_cost)
            sigma = 1 / (1 + (np.e ** ((-1) * (np.inner(X, theta_copy)))))
            sigma -= y
            Xt_mult_vec = np.transpose(X).dot(sigma)
            Xt_mult_vec *= self.eta
            theta_copy = theta_copy - Xt_mult_vec
            curr_cost = compute_cost(X, y, theta_copy)
        self.theta = theta_copy
        self.cost_list = J_history

    def predict(self, X):
        """Return the predicted class label"""
        pass














