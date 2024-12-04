import pytest
import random
import numpy as np

from src.binary_logistic_regression import BinaryLogisticRegression

# Sets random seed for testing purposes
random.seed(0)
np.random.seed(0)

# BINARY LOGISTIC REGRESSION
# test model with 1 predictor, batch size of 1 and conv threshold of 1e-2 (only 2 classes bc binary)
test_model1 = BinaryLogisticRegression(n_features=1, batch_size=1, epochs=100)

# test data with bias term
x_bias = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [1.2, 1]])  
# labels
y = np.array([0, 0, 1, 1, 1, 0])

# calc init loss
initial_loss = test_model1.loss(x_bias, y)
assert initial_loss == pytest.approx(0.693, 0.001)

# checking that weights have the correct shape
assert test_model1.weights.shape == (2,)

# train model
test_model1.train(x_bias, y)

# test model by inputting training data --> accuracy should be 100%
x_bias_test = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [1.2, 1]])
predictions = test_model1.predict(x_bias_test)
expected_predictions = np.array([0, 0, 1, 1, 1, 0])
assert np.all(predictions == expected_predictions)
accuracy = test_model1.accuracy(x_bias_test, expected_predictions)
assert accuracy == pytest.approx(1.0, 0.01)

# input new unseen testing data --> accuracy should also be 100%
x_bias_test = np.array([[1.5, 1], [3.5, 1]])
predictions = test_model1.predict(x_bias_test)
expected_predictions = np.array([0, 1]) 
assert np.all(predictions == expected_predictions)
accuracy = test_model1.accuracy(x_bias_test, expected_predictions)
assert accuracy == pytest.approx(1.0, 0.01)

# testing weight calculations manually [as implemented in the code]
weights = np.zeros(2)  
x_sample = np.array([1, 1])  
y_sample = 0 
z = np.dot(weights, x_sample)
prediction = 1 / (1 + np.exp(-z))
gradient = (prediction - y_sample) * x_sample
test_gradient = np.array([0.5, 0.5])  
assert gradient == pytest.approx(test_gradient, 0.01)

# testing case with one data point and testing to see behavior of weights
test_model2 = BinaryLogisticRegression(n_features=1, batch_size=1, epochs=10)
x_train = np.array([[1, 1]])  
y_train = np.array([1])       
test_model2.train(x_train, y_train)
assert test_model3.weights[0] > 0 # positive
assert test_model3.weights[1] > 0 # bias also positive