import pytest
import random
import numpy as np
import unittest
from src.binary_logistic_regression import BinaryLogisticRegression

# Sets random seed for testing purposes
random.seed(0)
np.random.seed(0)

class TestBinaryLogisticRegression(unittest.TestCase):
    def setUp(self):
        """Set up the model and training data."""
        self.model = BinaryLogisticRegression(n_features=1, batch_size=1, epochs=100)
        self.x_bias = np.array([
            [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [1.2, 1]
        ])
        self.y = np.array([0, 0, 1, 1, 1, 0])

    def test_initial_loss(self):
        """Test the initial loss calculation."""
        initial_loss = self.model.loss(self.x_bias, self.y)
        self.assertAlmostEqual(initial_loss, 0.693, places=3)

    def test_weights_shape(self):
        """Test that the weights have the correct shape."""
        self.assertEqual(self.model.weights.shape, (2,))

    def test_training_on_training_data(self):
        """Test that testing on training data results in perfect accuracy."""
        self.model.train(self.x_bias, self.y)
        predictions = self.model.predict(self.x_bias)
        np.testing.assert_array_equal(predictions, self.y)
        accuracy = self.model.accuracy(self.x_bias, self.y)
        self.assertAlmostEqual(accuracy, 1.0, places=2)

    def test_predictions_on_unseen_data(self):
        """Test predictions and accuracy on unseen data."""
        self.model.train(self.x_bias, self.y)
        x_bias_test = np.array([[1.5, 1], [3.5, 1]])
        expected_predictions = np.array([0, 1])
        predictions = self.model.predict(x_bias_test)
        np.testing.assert_array_equal(predictions, expected_predictions)
        accuracy = self.model.accuracy(x_bias_test, expected_predictions)
        self.assertAlmostEqual(accuracy, 1.0, places=2)

    def test_gradient_calculation(self):
        """Test manual gradient calculation."""
        weights = np.zeros(2)
        x_sample = np.array([1, 1])
        y_sample = 0
        z = np.dot(weights, x_sample)
        prediction = 1 / (1 + np.exp(-z))
        gradient = (prediction - y_sample) * x_sample
        expected_gradient = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=2)

    def test_training_on_single_data_point(self):
        """Test model behavior when trained on a single data point."""
        single_point_model = BinaryLogisticRegression(n_features=1, batch_size=1, epochs=10)
        x_train = np.array([[1, 1]])
        y_train = np.array([1])
        single_point_model.train(x_train, y_train)
        self.assertGreater(single_point_model.weights[0], 0)  # (+) feature weight
        self.assertGreater(single_point_model.weights[1], 0)  # (+) bias term

    def test_all_features_zero(self):
        """Test training with all features set to zero."""
        x_bias_zero = np.zeros((5, 2))
        y_zero = np.array([0, 0, 1, 1, 0])
        self.model.train(x_bias_zero, y_zero)
        predictions = self.model.predict(x_bias_zero)
        # sigmoid values should be 50% --> leading to output of 1 [based on code's tie breaking implementation]
        self.assertTrue(np.all(predictions == 1))

    def test_all_labels_zero(self):
        """Test training with all labels the same."""
        y_same = np.array([0, 0, 0, 0, 0, 0])
        self.model.train(self.x_bias, y_same)
        predictions = self.model.predict(self.x_bias)
        self.assertTrue(np.all(predictions == 0)) 
    
    def test_all_labels_one(self):
        """Test training with all labels the same."""
        y_same = np.array([1, 1, 1, 1, 1, 1])
        self.model.train(self.x_bias, y_same)
        predictions = self.model.predict(self.x_bias)
        self.assertTrue(np.all(predictions == 1)) 

    def test_large_feature_values(self):
        """Test training with very large feature values."""
        x_bias_large = np.array([[10, 1], [20, 1], [30, 1]])
        y_large = np.array([0, 1, 1])
        self.model.train(x_bias_large, y_large)
        predictions = self.model.predict(x_bias_large)
        np.testing.assert_array_equal(predictions, y_large)

    def test_prediction_probabilities(self):
        """Test the output of prediction probabilities."""
        self.model.train(self.x_bias, self.y)
        x_bias_test = np.array([[1.5, 1], [3.5, 1]])
        probabilities = self.model.predict_probs(x_bias_test)  # New function
        self.assertTrue(np.all(probabilities >= 0) and np.all(probabilities <= 1), 
                        "Probabilities are not in [0, 1] range.")
        predictions = np.where(probabilities >= 0.5, 1, 0)
        expected_predictions = np.array([0, 1])
        np.testing.assert_array_equal(predictions, expected_predictions)


if __name__ == "__main__":
    unittest.main()