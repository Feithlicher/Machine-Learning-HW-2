import numpy as np

data = np.array([[1, 2, 10], [3, 4, 12]])
vec1 = np.array([1, 7])
# print(vec1)
vec2 = np.array([4, 4])
# nate = np.column_stack((vec1, vec2))
# print(nate)
# X_folds = np.split(vec1, 5, axis=0)
# print(X_folds)
# print(type(X_folds))
# nate2 = np.random.uniform(1, 5)
# print(nate2)
#
# nate3 = np.random.uniform(vec1, vec2)
# print(nate3)
# print(vec1 - vec2)
#
#
# nate4 = np.zeros_like((mat[0])) + 4
#
# print(nate4)


w_init = np.zeros_like((data[0])) + (1 / 3)
cur_mu = data.mean(axis=0)
cur_std = data.std(axis=0)
rand_min_mu = cur_mu - (2 * cur_std)
rand_max_mu = cur_mu + (2 * cur_std)
mu_init = np.random.uniform(rand_min_mu, rand_max_mu)
std_init = [np.random.uniform(0, 3) for _ in range(data.shape[1])]
param_mat = np.column_stack((w_init, mu_init, std_init))

print(param_mat.shape)
print(param_mat)



# print(X_training.std(axis=0))
# print(X_training.shape)
# print(X_training[:, 0].mean())
# print(X_training[:, 1].mean())

# cur_mu = X_training.mean(axis=0)
# cur_std = X_training.std(axis=0)
# print("cur_mu: ", cur_mu)
# print("cur_std: ", cur_std)
# rand_min_mu = cur_mu - (2 * cur_std)
# rand_max_mu = cur_mu + (2 * cur_std)
# print(rand_min_mu)
# print(rand_max_mu)
# mu = np.random.uniform(rand_min_mu, rand_max_mu)
# w = [2, 2]
#
# print("mu", mu)
# std_init = [np.random.uniform(0, 3) for x in range(2)]
# print("std_init", std_init)
# print(np.column_stack((w, mu, std_init)))


# cur_mu = cur_mu / 2
# print(cur_mu)
# dim_ind_vec = [x + 1 for x in range(2)]
# cur_mu = cur_mu * dim_ind_vec
# print(dim_ind_vec)
# print(cur_mu)
