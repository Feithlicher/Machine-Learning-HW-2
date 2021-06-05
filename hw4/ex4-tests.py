import numpy as np
import pandas as pd

training_set = pd.read_csv('training_set.csv')
test_set = pd.read_csv('test_set.csv')
X_training, y_training = training_set[['x1', 'x2']].values, training_set['y'].values
X_test, y_test = test_set[['x1', 'x2']].values, test_set['y'].values



nate = X_training[:1000]
print(nate.shape)


# vec1 = np.array([1, 2])
# vec2 = np.insert(vec1, 0, 5)
# print(vec2)