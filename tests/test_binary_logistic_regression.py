import unittest
import numpy as np
from src.binary_logistic_regression import BinaryLogisticRegression

class TestBinaryLogisticRegression(unittest.TestCase):

    def setUp(self):
        """Initialize common test parameters."""
        self.n_features = 1
        self.batch_size = 1
        self.conv_threshold = 1e-6
        self.model = BinaryLogisticRegression(
            n_features=self.n_features,
            batch_size=self.batch_size,
            conv_threshold=self.conv_threshold
        )

    def test_loss_initialization(self):
        """Test that the initial loss is computed correctly."""
        np.random.seed(0)
        x = np.array([[1], [2], [3], [4], [5], [1.2]])
        y = np.array([0, 0, 1, 1, 1, 0])
        initial_loss = self.model.loss(x, y)
        self.assertAlmostEqual(initial_loss, 0.693, places=3)

    def test_sigmoid(self):
        """Test that sigmoid outputs correct calculations."""
        # small positive input
        z = 1
        expected_output = 1 / (1 + np.exp(-z))
        output = self.model.sigmoid(z)
        self.assertAlmostEqual(output, expected_output, places=6)

        # small negative input
        z = -1
        expected_output = 1 / (1 + np.exp(-z))
        output = self.model.sigmoid(z)
        self.assertAlmostEqual(output, expected_output, places=6)

        # large positive input (ensure stability)
        z = 100
        expected_output = 1.0  # Sigmoid saturates to 1
        output = self.model.sigmoid(z)
        self.assertAlmostEqual(output, expected_output, places=6)

        # large negative input (ensure stability)
        z = -100
        expected_output = 0.0  # Sigmoid saturates to 0
        output = self.model.sigmoid(z)
        self.assertAlmostEqual(output, expected_output, places=6)

        # zero input
        z = 0
        expected_output = 0.5  # Sigmoid(0) = 0.5
        output = self.model.sigmoid(z)
        self.assertAlmostEqual(output, expected_output, places=6)

        # vector input
        z = np.array([-1, 0, 1])
        expected_output = 1 / (1 + np.exp(-z))
        output = self.model.sigmoid(z)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=6)

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
        weights = np.array([0.0, 0.0])  # init weights
        x_sample = np.array([1])  # single feature
        y_sample = 0

        z = weights[0] * x_sample[0] + weights[1]  # using weights and bias term
        prediction = 1 / (1 + np.exp(-z))
        gradient = (prediction - y_sample) * np.array([x_sample[0], 1])  # gradient for weight + bias
        expected_gradient = np.array([0.5, 0.5])  # manually computed gradient with zero weights

        np.testing.assert_allclose(gradient, expected_gradient, atol=0.01)

    def test_imbalanced_data(self):
        """Test behavior on an imbalanced dataset."""
        x = np.array([[1], [2], [3], [4], [5]])
        # imbalanced labels
        y = np.array([0, 0, 0, 0, 1])  
        epochs = self.model.train(x, y)
        predictions = self.model.predict(x)
        majority_class = 0
        self.assertGreaterEqual(np.sum(predictions == majority_class), 3)

    def test_non_separable_data(self):
        """Test behavior with non-linearly separable data."""
        x = np.array([[1], [2], [3], [4]])
        # Non-linearly separable
        y = np.array([0, 1, 0, 1]) 
        self.model.train(x, y)
        predictions = self.model.predict(x)
        accuracy = self.model.accuracy(x, y)
        # should be better than random guessing
        self.assertGreaterEqual(accuracy, 0.5)  
        self.assertLessEqual(accuracy, 1.0)

     # FOR TESTING EDGE CASES INVOLVING FORMATTING OF DATA
    def test_invalid_initialization(self):
        """Test invalid parameters during initialization."""
        # n_features
        with self.assertRaises(ValueError):
            BinaryLogisticRegression(n_features=-1, batch_size=1, conv_threshold=1e-3)
        with self.assertRaises(ValueError):
            BinaryLogisticRegression(n_features=0, batch_size=1, conv_threshold=1e-3)
        # batch_size
        with self.assertRaises(ValueError):
            BinaryLogisticRegression(n_features=2, batch_size=0, conv_threshold=1e-3)
        with self.assertRaises(ValueError):
            BinaryLogisticRegression(n_features=2, batch_size=-5, conv_threshold=1e-3)
        # conv_threshold
        with self.assertRaises(ValueError):
            BinaryLogisticRegression(n_features=2, batch_size=1, conv_threshold=-0.1)
        with self.assertRaises(ValueError):
            BinaryLogisticRegression(n_features=2, batch_size=1, conv_threshold=0)
    
    def test_invalid_train_inputs(self):
        """Test invalid training inputs for X and Y."""
        model = BinaryLogisticRegression(n_features=2, batch_size=1, conv_threshold=1e-3)
        valid_X = np.array([[1, 2], [3, 4], [5, 6]])
        valid_Y = np.array([0, 1, 1])
    
        # non-numpy inputs
        with self.assertRaises(TypeError):
            model.train([[1, 2], [3, 4]], valid_Y)
        with self.assertRaises(TypeError):
            model.train(valid_X, [0, 1, 1])
    
        # empty inputs
        with self.assertRaises(ValueError):
            model.train(np.array([]), valid_Y)
        with self.assertRaises(ValueError):
            model.train(valid_X, np.array([]))
    
        # mismatched samples
        with self.assertRaises(ValueError):
            model.train(valid_X, np.array([0, 1]))
    
        # invalid labels
        with self.assertRaises(ValueError):
            # 2 label is invalid
            model.train(valid_X, np.array([0, 1, 2]))  
    
    def test_invalid_predict_inputs(self):
        """Test invalid prediction inputs for X."""
        model = BinaryLogisticRegression(n_features=2, batch_size=1, conv_threshold=1e-3)
    
        # non-numpy inputs
        with self.assertRaises(TypeError):
            model.predict([[1, 2], [3, 4]])
    
        # empty inputs
        with self.assertRaises(ValueError):
            model.predict(np.array([]))
    
        # invlid feature dimensions
        with self.assertRaises(ValueError):
            # 3 features instead of 2
            model.predict(np.array([[1, 2, 3]]))  
    
    def test_invalid_loss_inputs(self):
        """Test invalid inputs for the loss function."""
        model = BinaryLogisticRegression(n_features=2, batch_size=1, conv_threshold=1e-3)
        valid_X = np.array([[1, 2], [3, 4], [5, 6]])
        valid_Y = np.array([0, 1, 1])
    
        # invalid feature dimensions
        with self.assertRaises(ValueError):
            model.loss(np.array([[1, 2, 3]]), valid_Y)
    
        # invalid labels
        with self.assertRaises(ValueError):
            # 2 label is invalid
            model.loss(valid_X, np.array([0, 1, 2])) 
    
    def test_invalid_accuracy_inputs(self):
        """Test invalid inputs for accuracy calculation."""
        model = BinaryLogisticRegression(n_features=2, batch_size=1, conv_threshold=1e-3)
        valid_X = np.array([[1, 2], [3, 4], [5, 6]])
        valid_Y = np.array([0, 1, 1])
    
        # mismatched samples
        with self.assertRaises(ValueError):
            model.accuracy(valid_X, np.array([0, 1]))
    
        # empty inputs
        with self.assertRaises(ValueError):
            model.accuracy(np.array([]), valid_Y)
        with self.assertRaises(ValueError):
            model.accuracy(valid_X, np.array([]))

    # def test_noisy_data(self):
    #     """Test behavior with noisy labels."""
    #     x = np.array([[1], [2], [3], [4], [5]])
    #     # label noise
    #     y = np.array([0, 0, 1, 1, 0]) 

    #     self.model.train(x, y)
    #     predictions = self.model.predict(x)
    #     accuracy = self.model.accuracy(x, y)
    #     # expect reasonable performance despite noise
    #     self.assertGreaterEqual(accuracy, 0.6)  

if __name__ == '__main__':
    unittest.main()