import unittest
import numpy as np
from src.one_vs_all import OneVsAllLogisticRegression
from src.binary_logistic_regression import BinaryLogisticRegression


class TestOneVsAllLogisticRegression(unittest.TestCase):

    def setUp(self):
        """Initialize common test parameters."""
        self.n_classes = 3
        self.n_features = 2
        self.batch_size = 1
        self.model = OneVsAllLogisticRegression(
            n_classes=self.n_classes,
            binary_classifier_class=BinaryLogisticRegression,
            n_features=self.n_features,
            batch_size=self.batch_size,
            random_state=42
        )

    def test_train_creates_correct_classifiers(self):
        """Test that `train` creates one classifier per class and trains it correctly."""
        X = np.array([[1, 0], [0, 1], [-1, 0], [2, 0], [0,2], [-2,0]])  # Data separable
        Y = np.array([2, 1, 0, 2, 1, 0])  # 3 classes: 0, 1, 2

        # Train the model
        self.model.train(X, Y)

        # Check that the correct number of classifiers was created
        self.assertEqual(len(self.model.classifiers), self.n_classes)

        # Check that each classifier was trained with the correct binary labels
        for class_i, classifier in self.model.classifiers.items():
            binary_labels = np.where(Y == class_i, 1, 0)
            predictions = classifier.predict(X)
            np.testing.assert_array_equal(predictions, binary_labels)

    def test_predict_correct_classes(self):
        """Test that `predict` returns the correct class labels."""
        X = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
        Y = np.array([0, 1, 2, 1])  # 3 classes: 0, 1, 2

        # Train the model
        self.model.train(X, Y)

        # Predict class labels
        predictions = self.model.predict(X)

        # Check if predictions match true labels
        np.testing.assert_array_equal(predictions, Y)

    def test_accuracy_calculation(self):
        """Test that `accuracy` computes the correct value."""
        X = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
        Y = np.array([0, 1, 2, 1])  # 3 classes: 0, 1, 2

        # Train the model
        self.model.train(X, Y)

        # Calculate accuracy
        acc = self.model.accuracy(X, Y)

        # Expected accuracy: 100%
        self.assertAlmostEqual(acc, 1.0)

    def test_non_separable_data(self):
        """Test that the model handles non-linearly separable data."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        Y = np.array([0, 1, 0, 2]) 

        # Train the model
        self.model.train(X, Y)

        # Predict class labels
        predictions = self.model.predict(X)

        # Accuracy might not be perfect due to non-separability
        acc = self.model.accuracy(X, Y)
        self.assertGreaterEqual(acc, 0.5)  # At least better than random guessing

    def test_unseen_data(self):
        """Test that the model generalizes to unseen data."""
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

    # -----------------
    # Parameter Validation Tests
    # -----------------

    def test_invalid_n_classes(self):
        """Test that an invalid `n_classes` parameter raises an error."""
        with self.assertRaises(ValueError):
            OneVsAllLogisticRegression(
                n_classes=0,  # Invalid
                binary_classifier_class=BinaryLogisticRegression,
                n_features=2,
                batch_size=1
            )

    def test_invalid_binary_classifier_class(self):
        """Test that an invalid `binary_classifier_class` raises an error."""
        with self.assertRaises(TypeError):
            OneVsAllLogisticRegression(
                n_classes=3,
                binary_classifier_class="NotAClass",  # Invalid
                n_features=2,
                batch_size=1
            )

    def test_invalid_n_features(self):
        """Test that an invalid `n_features` parameter raises an error."""
        with self.assertRaises(ValueError):
            OneVsAllLogisticRegression(
                n_classes=3,
                binary_classifier_class=BinaryLogisticRegression,
                n_features=-1,  # Invalid
                batch_size=1
            )

    def test_invalid_batch_size(self):
        """Test that an invalid `batch_size` parameter raises an error."""
        with self.assertRaises(ValueError):
            OneVsAllLogisticRegression(
                n_classes=3,
                binary_classifier_class=BinaryLogisticRegression,
                n_features=2,
                batch_size=0  # Invalid

            )

    def test_invalid_epochs(self):
        """Test that an invalid `epochs` parameter raises an error."""
        with self.assertRaises(ValueError):
            OneVsAllLogisticRegression(
                n_classes=3,
                binary_classifier_class=BinaryLogisticRegression,
                n_features=2,
                batch_size=1,
                max_epochs=-10  # Invalid
            )

    def test_train_empty_data(self):
        """Test that training with empty data raises an error."""
        X = np.array([])
        Y = np.array([])
        with self.assertRaises(ValueError):
            self.model.train(X, Y)


    def test_predict_invalid_dimensions(self):
        """Test that `predict` with invalid dimensions raises an error."""
        X = np.array([[1, 2, 3]])  # More features than expected
        with self.assertRaises(ValueError):
            self.model.predict(X)
    
    def test_train_dimension_mismatch(self):
        """Test that training with mismatched dimensions raises an error."""
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([0])  # Mismatch in size
        with self.assertRaises(ValueError):
            self.model.train(X, Y)
