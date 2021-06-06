import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

training_set = pd.read_csv('training_set.csv')
test_set = pd.read_csv('test_set.csv')
X_training, y_training = training_set[['x1', 'x2']].values, training_set['y'].values
X_test, y_test = test_set[['x1', 'x2']].values, test_set['y'].values
# np.random.seed(42)


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
        self.cost_list = [] # a list of the costs that you've calculated in each iteration

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
        # bias trick
        ones_col = np.ones(X.shape[0])
        X = np.column_stack((ones_col, X))

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
        # bias trick
        if len(X.shape) > 1:
            ones_col = np.ones(X.shape[0])
            X = np.column_stack((ones_col, X))
        else:
            X = np.insert(X, 0, 1)

        temp_ans = (1 + (np.e ** ((-1) * (np.inner(self.theta, X)))))
        sigma = 1 / temp_ans
        return np.array([1 if sig > 0.5 else 0 for sig in sigma])


# calc normal pdf
def norm_pdf(data, mu, sigma):
    exp = (((data - mu) / sigma) ** 2) / (-2)
    base_inv = sigma * ((2 * np.pi) ** 0.5)
    return (np.e ** exp) / base_inv


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.dim_gaussians_dict = {}
        self.responsibilities_dict = {}
        self.cost_list = []  # a list of the costs that you've calculated in each iteration

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params:
        """
        for dim in range(data.shape[1]):
            w_init = np.array([0 for _ in range(self.k)]) + (1 / self.k)
            cur_mu = data[:, dim].mean()
            cur_std = data[:, dim].std()
            rand_min_mu = cur_mu - (2 * cur_std)
            rand_max_mu = cur_mu + (2 * cur_std)
            mu_init = [np.random.uniform(rand_min_mu, rand_max_mu) for _ in range(self.k)]
            std_init = [np.random.uniform(0, 3) for _ in range(self.k)]
            self.dim_gaussians_dict[dim] = np.column_stack((w_init, mu_init, std_init))

    def compute_gmm_cost(self, data, dim):
        responsibilities_mat = np.array([norm_pdf(point, self.dim_gaussians_dict[dim][:, 1], self.dim_gaussians_dict[dim][:, 2]) for point in data[:, dim]])
        responsibilities_mat = responsibilities_mat * self.dim_gaussians_dict[dim][:, 0]
        responsibilities_mat = np.log(np.sum(responsibilities_mat, axis=1))
        return np.sum(responsibilities_mat) * (-1)

    def expectation(self, data, dim):
        """
        E step - calculating responsibilities
        """
        responsibilities_mat = np.array([norm_pdf(point, self.dim_gaussians_dict[dim][:, 1], self.dim_gaussians_dict[dim][:, 2]) for point in data[:, dim]])
        responsibilities_mat = responsibilities_mat * self.dim_gaussians_dict[dim][:, 0]
        responsibilities_vec_sum_axis1 = np.sum(responsibilities_mat, axis=1)
        self.responsibilities_dict[dim] = np.transpose((np.transpose(responsibilities_mat) / responsibilities_vec_sum_axis1))

    def maximization(self, data, dim):
        """
        M step - updating distribution params
        """
        # updating weights
        responsibilities_avg_vec = np.sum(self.responsibilities_dict[dim], axis=0)
        self.dim_gaussians_dict[dim][:, 0] = responsibilities_avg_vec / data.shape[0]

        # updating means
        mu_sum = np.sum(np.transpose((np.transpose(self.responsibilities_dict[dim])) * data[:, dim]), axis=0)
        self.dim_gaussians_dict[dim][:, 1] = mu_sum / responsibilities_avg_vec

        # updating stds
        x_i_columns = np.transpose(np.transpose(np.zeros_like(self.responsibilities_dict[dim])) + data[:, dim])
        std_sum = np.sum(self.responsibilities_dict[dim] * ((x_i_columns - self.dim_gaussians_dict[dim][:, 1]) ** 2), axis=0)
        self.dim_gaussians_dict[dim][:, 2] = ((std_sum / responsibilities_avg_vec) ** 0.5)

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        for dim in self.dim_gaussians_dict.keys():
            J_history = []
            k = 0
            curr_cost = self.compute_gmm_cost(data, dim)
            J_history.append(0)
            while (abs(curr_cost - J_history[len(J_history) - 1]) > self.eps) and (k < self.n_iter):
                if k == 0:
                    J_history.pop()
                k += 1
                J_history.append(curr_cost)
                self.expectation(data=data, dim=dim)
                self.maximization(data=data, dim=dim)
                curr_cost = self.compute_gmm_cost(data, dim)

    def get_dist_params(self):
        return self.dim_gaussians_dict


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1):
        self.k = k
        self.cls0_prior = 0
        self.cls1_prior = 0
        self.cls0_gaussians_dict = {}
        self.cls1_gaussians_dict = {}

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        # training classes partition
        dataset = np.column_stack((X, y))
        cls0 = np.array([vec for vec in dataset if vec[-1] == 0])
        cls1 = np.array([vec for vec in dataset if vec[-1] == 1])
        cls0 = cls0[:, 0:2]
        cls1 = cls1[:, 0:2]

        # get priors
        self.cls0_prior = cls0.shape[0] / dataset.shape[0]
        self.cls1_prior = cls1.shape[0] / dataset.shape[0]

        # get classes distribution parameters
        cls0_em = EM(self.k)
        cls1_em = EM(self.k)
        cls0_em.fit(cls0)
        cls1_em.fit(cls1)
        self.cls0_gaussians_dict = cls0_em.get_dist_params()
        self.cls1_gaussians_dict = cls1_em.get_dist_params()

    def get_likelihood(self, X, cls_gaussians_dict):
        dim_pdf_vec = []
        for dim in cls_gaussians_dict.keys():
            if len(X.shape) > 1:
                pdf_mat = np.array([norm_pdf(point, cls_gaussians_dict[dim][:, 1], cls_gaussians_dict[dim][:, 2]) for point in X[:, dim]])
            else:
                pdf_mat = np.array([norm_pdf(X, cls_gaussians_dict[dim][:, 1], cls_gaussians_dict[dim][:, 2])])
            pdf_mat = pdf_mat * cls_gaussians_dict[dim][:, 0]
            dim_pdf_vec.append(np.sum(pdf_mat, axis=1))
        return dim_pdf_vec[0] * dim_pdf_vec[1]

    def predict(self, X):
        """Return the predicted class label"""
        cls0_post = self.get_likelihood(X, self.cls0_gaussians_dict) * self.cls0_prior
        cls1_post = self.get_likelihood(X, self.cls1_gaussians_dict) * self.cls1_prior
        bool_ind_vec = cls1_post > cls0_post
        return np.where(bool_ind_vec==False, 0, bool_ind_vec)


# Model Evaluation
first1000_X_trainig = np.array(X_training[:1000])
first1000_y_training = np.array(y_training[:1000])
first500_X_test = np.array(X_test[:500])
first500_y_test = np.array(y_test[:500])

# first 1000            START
# LoR model eval
first1000_LoR = LogisticRegressionGD(theta_shape=3)
first1000_LoR.fit(first1000_X_trainig, first1000_y_training)

# Compute Lor training Accuracy - first 1000
first1000_LoR_pred_training = first1000_LoR.predict(first1000_X_trainig)
first1000_LoR_training_accuracy = 0
diff_vec_train = first1000_y_training - first1000_LoR_pred_training
for diff in diff_vec_train:
    if diff == 0:
        first1000_LoR_training_accuracy += 1
first1000_Lor_training_accuracy = first1000_LoR_training_accuracy / 1000
print("first 1000 LoR training accuracy: ", first1000_Lor_training_accuracy)

# Compute Lor test Accuracy - first 500
first500_LoR_pred_test = first1000_LoR.predict(first500_X_test)
first500_LoR_test_accuracy = 0
diff_vec_test = first500_y_test - first500_LoR_pred_test
for diff in diff_vec_test:
    if diff == 0:
        first500_LoR_test_accuracy += 1
first500_LoR_test_accuracy = first500_LoR_test_accuracy / 500
print("first 500 LoR test accuracy: ", first500_LoR_test_accuracy)


# NB model eval - first 1000/500
nbg = NaiveBayesGaussian(k=2)
nbg.fit(first1000_X_trainig, first1000_y_training)

pred_vec = nbg.predict(first1000_X_trainig)
diff_vec = first1000_y_training - pred_vec
diff_count = 0
for diff in diff_vec:
    if diff == 0:
        diff_count += 1
first1000_NB_trainig_accuracy = diff_count / 1000
print("first 1000 samples NB training accuracy: ", first1000_NB_trainig_accuracy)

pred_vec = nbg.predict(first500_X_test)
diff_vec = first500_y_test - pred_vec
diff_count = 0
for diff in diff_vec:
    if diff == 0:
        diff_count += 1
first500_NB_test_accuracy = diff_count / 500
print("first 500 samples NB test accuracy: ", first500_NB_test_accuracy)

# Cost Vs the iteration number LoR - first 1000
plt.plot(np.arange(len(first1000_LoR.cost_list)), first1000_LoR.cost_list)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost as a function of iterations - first 1000')
plt.show()
# first 1000            END






# all features            START
# LoR model eval
all_LoR = LogisticRegressionGD(theta_shape=3)
all_LoR.fit(X_training, y_training)

# Compute Lor training Accuracy - all features
all_LoR_pred_training = all_LoR.predict(X_training)
all_LoR_training_accuracy = 0
diff_vec_train = y_training - all_LoR_pred_training
for diff in diff_vec_train:
    if diff == 0:
        all_LoR_training_accuracy += 1
all_Lor_training_accuracy = all_LoR_training_accuracy / X_training.shape[0]
print("All features LoR training accuracy: ", all_Lor_training_accuracy)

# Compute Lor test Accuracy - all features
all_LoR_pred_test = all_LoR.predict(X_test)
all_LoR_test_accuracy = 0
diff_vec_test = y_test - all_LoR_pred_test
for diff in diff_vec_test:
    if diff == 0:
        all_LoR_test_accuracy += 1
all_LoR_test_accuracy = all_LoR_test_accuracy / X_test.shape[0]
print("All features LoR test accuracy: ", all_LoR_test_accuracy)


# NB model eval - all features
all_nbg = NaiveBayesGaussian(k=2)
all_nbg.fit(X_training, y_training)

pred_vec = all_nbg.predict(X_training)
diff_vec = y_training - pred_vec
diff_count = 0
for diff in diff_vec:
    if diff == 0:
        diff_count += 1
all_NB_trainig_accuracy = diff_count / X_training.shape[0]
print("All features samples NB training accuracy: ", all_NB_trainig_accuracy)

pred_vec = all_nbg.predict(X_test)
diff_vec = y_test - pred_vec
diff_count = 0
for diff in diff_vec:
    if diff == 0:
        diff_count += 1
all_NB_test_accuracy = diff_count / X_test.shape[0]
print("All features samples NB test accuracy: ", all_NB_test_accuracy)

# Cost Vs the iteration number LoR - all features
plt.plot(np.arange(len(all_LoR.cost_list)), all_LoR.cost_list)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost as a function of iterations - all features')
plt.show()
# all features            END





















# lor = LogisticRegressionGD(theta_shape=3)
# lor.fit(X_training, y_training)
# nate2 = lor.predict(X_test)
# nate3 = np.column_stack((y_test, nate2))
# print()



# nbg = NaiveBayesGaussian(k=1)
# nbg.fit(X_training, y_training)
# one_inst = X_test[0]
# # res = nbg.get_likelihood(one_inst, nbg.cls0_gaussians_dict)
# res2 = nbg.predict(X_test)
# nat = np.column_stack((y_test, res2))
# print(res2)



# em = EM(k=2)
# em.fit(X_training)
# print(em.get_dist_params())


# em.init_params(X_training)
# em.expectation(X_training)
# nate = em.responsibilities_dict[0]
# print(em.responsibilities_dict[0])
# print(em.responsibilities_dict[0].shape)
# print(np.sum(nate, axis=1))
# print("SPACE SPACE")
# print("SPACE SPACE")
# print("em.dim_gaussians_dict[0]")
# print(em.dim_gaussians_dict[0])
# print("SPACE SPACE")
# print("SPACE SPACE")
#
# em.maximization(X_training)
# print(em.dim_gaussians_dict)
# print(em.compute_gmm_cost(X_training))




# dataset = np.column_stack((X_training, y_training))
# cls0 = np.array([vec for vec in dataset if vec[-1] == 0])
# cls1 = np.array([vec for vec in dataset if vec[-1] == 1])
# cls0 = cls0[:, 0:2]
# cls1 = cls1[:, 0:2]
#
# # cls0_em = EM(2)
# cls1_em = EM(2)
# # cls0_em.fit(cls0)
# cls1_em.fit(cls1)
# print()