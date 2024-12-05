import unittest
import numpy as np
import random
from src.all_pairs import AllPairsLogisticRegression
from src.binary_logistic_regression import BinaryLogisticRegression

random.seed(0)
np.random.seed(0)

class TestAllPairsLogisticRegression(unittest.TestCase):
    def setUp(self):
        """Set up the model and training data."""
        self.model = AllPairsLogisticRegression(
            n_classes=3,
            n_features=2,
            batch_size=1,
            epochs=100,
            binary_classifier_class=BinaryLogisticRegression
        )
        self.X_train = np.array([
            [1, 2], [2, 1], [2, 2],  
            [3, 4], [4, 3], [4, 4],  
            [5, 6], [6, 5], [6, 6]   
        ])
        self.Y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.model.train(self.X_train, self.Y_train)

    def test_all_classifiers_trained(self):
        """Test that all required binary classifiers are trained."""
        unique_pairs = [(i, j) for i in range(3) for j in range(3) if i < j]
        self.assertEqual(len(self.model.classifiers), len(unique_pairs))

    def test_predict(self):
        """Test the predict function on training data."""
        predictions = self.model.predict(self.X_train)
        np.testing.assert_array_equal(predictions, self.Y_train)

    def test_accuracy(self):
        """Test the accuracy calculation on training data."""
        accuracy = self.model.accuracy(self.X_train, self.Y_train)
        self.assertAlmostEqual(accuracy, 1.0, places=2)

    def test_predict_on_unseen_data(self):
        """Test predictions on unseen testing data."""
        X_test = np.array([
            [2, 3],  
            [3.5, 3.5],  
            [5.5, 5.5]  
        ])
        expected_predictions = np.array([0, 1, 2])
        predictions = self.model.predict(X_test)
        np.testing.assert_array_equal(predictions, expected_predictions)


if __name__ == "__main__":
    unittest.main()
