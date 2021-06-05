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

mat = np.array([[1, 2, 3], [4, 5, 6]])
vec1 = np.array([1, 5, 3])
vec2 = np.array([4, 1, 6])
bool_vec = vec1 > vec2
print(bool_vec)
bool_vec = np.where(bool_vec==False, 0, bool_vec)
print(bool_vec)