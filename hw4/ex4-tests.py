import numpy as np
import pandas as pd

training_set = pd.read_csv('training_set.csv')
test_set = pd.read_csv('test_set.csv')
X_training, y_training = training_set[['x1', 'x2']].values, training_set['y'].values
X_test, y_test = test_set[['x1', 'x2']].values, test_set['y'].values

# mat - rows are linearly dependent
mat = np.array([[1, 2], [2, 4]])
for i in range(1998):
    mat = np.row_stack((mat, (mat[(i + 1)] * (1.001))))
zero_lebals = np.array([0 for _ in range(500)])
one_lebals = np.array([1 for _ in range(500)])
labels = np.concatenate((zero_lebals, one_lebals, zero_lebals, one_lebals))
mat = np.column_stack((mat, labels))
mat = pd.DataFrame(mat)
mat.columns = ["x1", "x2", "y"]

print(mat)
