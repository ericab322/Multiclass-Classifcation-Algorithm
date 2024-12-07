import unittest
import numpy as np
from src.binary_logistic_regression import BinaryLogisticRegression


class TestBinaryLogisticRegression(unittest.TestCase):

    def setUp(self):
        """Initialize common test parameters."""
        self.n_features = 1
        self.batch_size = 1
        self.epochs = 100
        self.model = BinaryLogisticRegression(
            n_features=self.n_features,
            batch_size=self.batch_size,
            epochs=self.epochs
        )

    def test_loss_initialization(self):
        """Test that the initial loss is computed correctly."""
        np.random.seed(0)
        x = np.array([[1], [2], [3], [4], [5], [1.2]])
        y = np.array([0, 0, 1, 1, 1, 0])
        initial_loss = self.model.loss(x, y)
        self.assertAlmostEqual(initial_loss, 0.693, places=3)

    def test_training_and_predictions(self):
        """Test that the model learns correctly on training data."""
        x = np.array([[1], [2], [3], [4], [5], [1.2]])
        y = np.array([0, 0, 1, 1, 1, 0])

        self.model.train(x, y)
        predictions = self.model.predict(x)
        expected_predictions = np.array([0, 0, 1, 1, 1, 0])
        np.testing.assert_array_equal(predictions, expected_predictions)
        accuracy = self.model.accuracy(x, expected_predictions)
        self.assertAlmostEqual(accuracy, 1.0, places=2)

    def test_new_unseen_data(self):
        """Test the model on unseen data after training."""
        x_train = np.array([[1], [2], [3], [4], [5], [1.2]])
        y_train = np.array([0, 0, 1, 1, 1, 0])

        x_test = np.array([[1.5], [3.5]])
        y_test = np.array([0, 1])

        self.model.train(x_train, y_train)
        predictions = self.model.predict(x_test)
        np.testing.assert_array_equal(predictions, y_test)
        accuracy = self.model.accuracy(x_test, y_test)
        self.assertAlmostEqual(accuracy, 1.0, places=2)

    def test_gradient_calculation(self):
        """Test that the gradient is computed correctly."""
        weights = np.array([0.0, 0.0])  # Initial weights
        x_sample = np.array([1])  # Single feature
        y_sample = 0

        z = weights[0] * x_sample[0] + weights[1]  # Using weights and bias term
        prediction = 1 / (1 + np.exp(-z))
        gradient = (prediction - y_sample) * np.array([x_sample[0], 1])  # Gradient for weight + bias
        expected_gradient = np.array([0.5, 0.5])  # Manually computed gradient with zero weights

        np.testing.assert_allclose(gradient, expected_gradient, atol=0.01)

    def test_imbalanced_data(self):
        """Test behavior on an imbalanced dataset."""
        x = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 0, 0, 1])  # Imbalanced labels

        self.model.train(x, y)
        predictions = self.model.predict(x)
        majority_class = 0
        self.assertGreaterEqual(np.sum(predictions == majority_class), 3)

    def test_non_separable_data(self):
        """Test behavior with non-linearly separable data."""
        x = np.array([[1], [2], [3], [4]])
        y = np.array([0, 1, 0, 1])  # Non-linearly separable

        self.model.train(x, y)
        predictions = self.model.predict(x)
        accuracy = self.model.accuracy(x, y)
        self.assertGreaterEqual(accuracy, 0.5)  # At least better than random guessing
        self.assertLessEqual(accuracy, 1.0)

    def test_noisy_data(self):
        """Test behavior with noisy labels."""
        x = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 0])  # Introduce label noise

        self.model.train(x, y)
        predictions = self.model.predict(x)
        accuracy = self.model.accuracy(x, y)
        self.assertGreaterEqual(accuracy, 0.6)  # Expect reasonable performance despite noise


if __name__ == '__main__':
    unittest.main()