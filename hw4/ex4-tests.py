import numpy as np
import pandas as pd

training_set = pd.read_csv('training_set.csv')
test_set = pd.read_csv('test_set.csv')
X_training, y_training = training_set[['x1', 'x2']].values, training_set['y'].values
X_test, y_test = test_set[['x1', 'x2']].values, test_set['y'].values
np.random.seed(42)



# calc normal pdf
# def norm_pdf(data, mu, sigma):
#     exp = (((data - mu) / sigma) ** 2) / (-2)
#     base_inv = sigma * ((2 * np.pi) ** 0.5)
#     return (np.e ** exp) / base_inv
#
#
# print(norm_pdf(1, np.array([1, 2]), np.array([3, 4])))
vec1 = np.array([1, 2])
vec = np.array([5, 6, 7])
mat = np.array([[1, 2], [3, 4], [5, 3]])
mat2 = np.array([[7, 2], [3, 4]])
print(mat)
print("SPACE")
# print(np.transpose(np.transpose(mat) * vec))
# nate = vec.reshape((1, vec.shape[0]))
# print("nate", nate)
# print(np.column_stack((vec, vec)))
nate = np.transpose(np.transpose(np.zeros_like(mat)) + vec)
print(np.transpose(np.transpose(np.zeros_like(mat)) + vec))
print("SPACE")
print((nate - vec1))
print((nate - vec1) ** 2)
nate2 = (nate - vec1) ** 2
print("SPACE")
print(mat * nate2)

















# print("param_mat", param_mat)

# k = 2
# dim_dict = {}
# for dim in range(X_training.shape[1]):
#     w_init = np.array([0 for _ in range(k)]) + (1 / k)
#     cur_mu = X_training[:, dim].mean()
#     cur_std = X_training[:, dim].std()
#     rand_min_mu = cur_mu - (2 * cur_std)
#     rand_max_mu = cur_mu + (2 * cur_std)
#     mu_init = [np.random.uniform(rand_min_mu, rand_max_mu) for _ in range(k)]
#     std_init = [np.random.uniform(0, 3) for _ in range(k)]
#     dim_dict[dim] = np.column_stack((w_init, mu_init, std_init))
# #
# #
# #
# print(dim_dict)
# print(dim_dict[0][:, 1])

# dim0 = X_training[:, 0]
# nate = norm_pdf(X_training[:, 0], np.array([1, 2]), np.array([3, 4]))
# print(nate)
# print(norm_pdf(1, 1, 3))
# print(norm_pdf(1, 2, 4))
#
# dim0_porbs = [norm_pdf(point) for point in X_training[:, 0]]
#
#
#
# dan = np.array([1, 2, 3])
# print(dan[1:3])