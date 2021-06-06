import numpy as np
import pandas as pd

training_set = pd.read_csv('training_set.csv')
test_set = pd.read_csv('test_set.csv')
X_training, y_training = training_set[['x1', 'x2']].values, training_set['y'].values
X_test, y_test = test_set[['x1', 'x2']].values, test_set['y'].values

# row_dependent_mat
row_dependent_mat = np.array([[1, 2], [2, 4]])
for i in range(1998):
    row_dependent_mat = np.row_stack((row_dependent_mat, (row_dependent_mat[(i + 1)] * (1.001))))
zero_lebals = np.array([0 for _ in range(500)])
one_lebals = np.array([1 for _ in range(500)])
labels = np.concatenate((zero_lebals, one_lebals, zero_lebals, one_lebals))
row_dependent_mat = np.column_stack((row_dependent_mat, labels))
row_dependent_mat = pd.DataFrame(row_dependent_mat)
row_dependent_mat.columns = ["x1", "x2", "y"]

# column_dependent_mat
left_column = np.array([i for i in range(1, 2001)])
right_column = 2 * left_column
column_dependent_mat = np.column_stack((left_column, right_column))
column_dependent_mat = np.column_stack((column_dependent_mat, labels))
column_dependent_mat = pd.DataFrame(column_dependent_mat)
column_dependent_mat.columns = ["x1", "x2", "y"]

print(column_dependent_mat)
