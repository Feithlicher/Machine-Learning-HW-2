import numpy as np
import pandas as pd

training_set = pd.read_csv('training_set.csv')
test_set = pd.read_csv('test_set.csv')
X_training, y_training = training_set[['x1', 'x2']].values, training_set['y'].values
X_test, y_test = test_set[['x1', 'x2']].values, test_set['y'].values
np.random.seed(42)



# calc normal pdf
def norm_pdf(data, mu, sigma):
    exp = (((data - mu) / sigma) ** 2) / (-2)
    base_inv = sigma * ((2 * np.pi) ** 0.5)
    return (np.e ** exp) / base_inv


# print("param_mat", param_mat)

k = 2
dim_dict = {}
for dim in range(X_training.shape[1]):
    w_init = np.array([0 for _ in range(k)]) + (1 / k)
    cur_mu = X_training[:, dim].mean()
    cur_std = X_training[:, dim].std()
    rand_min_mu = cur_mu - (2 * cur_std)
    rand_max_mu = cur_mu + (2 * cur_std)
    mu_init = [np.random.uniform(rand_min_mu, rand_max_mu) for _ in range(k)]
    std_init = [np.random.uniform(0, 3) for _ in range(k)]
    dim_dict[dim] = np.column_stack((w_init, mu_init, std_init))



print(dim_dict)