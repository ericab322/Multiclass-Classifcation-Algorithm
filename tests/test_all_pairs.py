import unittest
import numpy as np
import random
from src.all_pairs import AllPairsLogisticRegression
from src.binary_logistic_regression import BinaryLogisticRegression

random.seed(0)
np.random.seed(0)

class TestAllPairsLogisticRegression(unittest.TestCase):
    def setUp(self):
        """Initialize common test parameters."""
        self.n_classes = 3
        self.n_features = 2
        self.batch_size = 1
        self.model = AllPairsLogisticRegression(
            n_classes=self.n_classes,
            binary_classifier_class=BinaryLogisticRegression,
            n_features=self.n_features,
            batch_size=self.batch_size,
            random_state=42
        )


    def test_all_classifiers_trained(self):
        """Test that all required binary classifiers are trained."""
        X = np.array([[1, 0], [0, 1], [-1, 0], [2, 0]])  # Data separable
        Y = np.array([2, 1, 0, 2])  # 3 classes: 0, 1, 2
        self.model.train(X, Y)
        unique_pairs = [(i, j) for i in range(3) for j in range(3) if i < j]
        self.assertEqual(len(self.model.classifiers), len(unique_pairs))
        
    def test_train_function(self):
        """Test the train function."""
        X = np.array([[1, 0], [0, 1], [-1, 0], [2, 0]])  # Data separable
        Y = np.array([2, 1, 0, 2])  # 3 classes: 0, 1, 2
        self.model.train(X, Y)
        self.assertTrue(len(self.model.classifiers) > 0)


    def test_train_creates_correct_classifiers_all_pairs(self):
        """Test that `train` creates one classifier for each pair of classes and trains it correctly."""
        X = np.array([[1, 0], [0, 1], [-1, 0], [2, 0]])  # Data separable
        Y = np.array([2, 1, 0, 2])  # 3 classes: 0, 1, 2

        # Train the model
        self.model.train(X, Y)

        # Check that the correct number of classifiers was created
        n_classes = len(np.unique(Y))
        expected_classifiers = n_classes * (n_classes - 1) // 2  # Number of class pairs
        self.assertEqual(len(self.model.classifiers), expected_classifiers)

        # Check each classifier's training data and predictions
        for class_i in range(n_classes):
            for class_j in range(class_i + 1, n_classes):
                # Get the binary classifier for this class pair
                classifier = self.model.classifiers[(class_i, class_j)]

                # Filter data for classes class_i and class_j
                mask = (Y == class_i) | (Y == class_j)
                X_pair = X[mask]
                Y_pair = Y[mask]

                # Convert labels to binary: class_i -> 1, class_j -> 0
                binary_labels = np.where(Y_pair == class_i, 1, 0)

                # Ensure the classifier's predictions match the binary labels
                predictions = classifier.predict(X_pair)
                np.testing.assert_array_equal(predictions, binary_labels)


    def test_accuracy(self):
        """Test the accuracy calculation on training data."""
        X = np.array([[1, 0], [0, 1], [-1, 0], [2, 0]])  # Data separable
        Y = np.array([2, 1, 0, 2])  # 3 classes: 0, 1, 2

        # Train the model
        self.model.train(X, Y)
        accuracy = self.model.accuracy(X, Y)
        self.assertAlmostEqual(accuracy, 1.0, places=2)

    def test_predict_on_unseen_data(self):
        """Test predictions on unseen testing data."""
        X_train = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
        Y_train = np.array([0, 1, 2, 1])  # 3 classes: 0, 1, 2

        X_test = np.array([[2, 2], [-2, -2], [3, 4], [8, -10]])
        Y_test = np.array([0, 1, 0, 2])  # Test on similar data

        # Train the model
        self.model.train(X_train, Y_train)

        # Predict on unseen data
        predictions = self.model.predict(X_test)

        # Check if predictions match true labels
        np.testing.assert_array_equal(predictions, Y_test)


    def test_train_empty_data(self):
        """Test that training with empty data raises an error."""
        X_empty = np.array([])
        Y_empty = np.array([])
        with self.assertRaises(ValueError):
            self.model.train(X_empty, Y_empty)
            
        
    def test_train_dimension_mismatch(self):
        """Test that training with mismatched dimensions raises an error."""
        X_mismatch = np.array([[1, 2], [3, 4]])  
        Y_mismatch = np.array([0])  
        with self.assertRaises(ValueError):
            self.model.train(X_mismatch, Y_mismatch)

    def test_invalid_binary_classifier_class(self):
        """Test that an invalid `binary_classifier_class` raises an error."""
        with self.assertRaises(TypeError):
            AllPairsLogisticRegression(
                n_classes=3,
                n_features=2,
                batch_size=1,
                epochs=100,
                binary_classifier_class="NotAClass",
                random_state=42
            )
            
    def test_non_separable_data(self):
        """Test the model on non-linearly separable data."""
        X_non_separable = np.array([
            [1, 2], [2, 1], [2, 2],
            [3, 4], [4, 3], [4, 4],
            [5, 6], [6, 5], [6, 6]
        ])
        Y_non_separable = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]) 
        self.model.train(X_non_separable, Y_non_separable)
        accuracy = self.model.accuracy(X_non_separable, Y_non_separable)
        self.assertGreaterEqual(accuracy, 0.5)  
    
    # # -----------------
    # # Additional Tests
    # # -----------------

    def test_predict_correct_classes(self):
        """Test that `predict` returns the correct class labels."""
        X = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
        Y = np.array([0, 1, 2, 1]) 

        self.model.train(X, Y)

        predictions = self.model.predict(X)

        np.testing.assert_array_equal(predictions, Y)

    def test_accuracy_calculation(self):
        """Test that `accuracy` computes the correct value."""
        X = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
        Y = np.array([0, 1, 2, 1])  

        self.model.train(X, Y)

        acc = self.model.accuracy(X, Y)

        self.assertAlmostEqual(acc, 1.0)

    def test_invalid_n_classes(self):
        """Test that an invalid `n_classes` parameter raises an error."""
        with self.assertRaises(ValueError):
            AllPairsLogisticRegression(
                n_classes=0,  # Invalid
                binary_classifier_class=BinaryLogisticRegression,
                n_features=2,
                batch_size=1,
                random_state=42
            )

    def test_invalid_n_features(self):
        """Test that an invalid `n_features` parameter raises an error."""
        with self.assertRaises(ValueError):
            AllPairsLogisticRegression(
                n_classes=3,
                binary_classifier_class=BinaryLogisticRegression,
                n_features=-1,  # Invalid
                batch_size=1,
                random_state=42
            )

    def test_invalid_batch_size(self):
        """Test that an invalid `batch_size` parameter raises an error."""
        with self.assertRaises(ValueError):
            AllPairsLogisticRegression(
                n_classes=3,
                binary_classifier_class=BinaryLogisticRegression,
                n_features=2,
                batch_size=0,  # Invalid
                random_state=42
            )

    def test_invalid_epochs(self):
        """Test that an invalid `max_epochs` parameter raises an error."""
        with self.assertRaises(ValueError):
            AllPairsLogisticRegression(
                n_classes=3,
                binary_classifier_class=BinaryLogisticRegression,
                n_features=2,
                batch_size=1,
                max_epochs=-10,  # Invalid
                random_state=42
            )

    def test_train_empty_data(self):
        """Test that training with empty data raises an error."""
        X = np.array([])
        Y = np.array([])
        with self.assertRaises(ValueError):
            self.model.train(X, Y)


    def test_predict_invalid_dimensions(self):
        """Test that `predict` with invalid dimensions raises an error."""
        X = np.array([[1, 2, 3]])  
        with self.assertRaises(ValueError):
            self.model.predict(X)
    
    def test_train_dimension_mismatch(self):
        """Test that training with mismatched dimensions raises an error."""
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([0])  # size mismatch
        with self.assertRaises(ValueError):
            self.model.train(X, Y)

if __name__ == "__main__":
    unittest.main()
