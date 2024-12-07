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
            binary_classifier_class=BinaryLogisticRegression,
            random_state=42
        )
        self.X_train = np.array([
            [1, 2], [2, 1], [2, 2],  
            [3, 4], [4, 3], [4, 4],  
            [5, 6], [6, 5], [6, 6]   
        ])
        self.Y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.X_train = np.hstack([np.ones((self.X_train.shape[0], 1)), self.X_train])

    def test_all_classifiers_trained(self):
        """Test that all required binary classifiers are trained."""
        unique_pairs = [(i, j) for i in range(3) for j in range(3) if i < j]
        self.assertEqual(len(self.model.classifiers), len(unique_pairs))
        
    def test_train_function(self):
        """Test the train function."""
        self.model.train(self.X_train, self.Y_train)
        self.assertTrue(len(self.model.classifiers) > 0)

    def test_predict(self):
        """Test the predict function on training data."""
        self.model.train(self.X_train, self.Y_train)
        predictions = self.model.predict(self.X_train)
        np.testing.assert_array_equal(predictions, self.Y_train)

    def test_accuracy(self):
        """Test the accuracy calculation on training data."""
        self.model.train(self.X_train, self.Y_train)
        accuracy = self.model.accuracy(self.X_train, self.Y_train)
        self.assertAlmostEqual(accuracy, 1.0, places=2)

    def test_predict_on_unseen_data(self):
        """Test predictions on unseen testing data."""
        X_raw_test = np.array([
            [2, 3],  
            [3.5, 3.5],  
            [5.5, 5.5]  
        ])
        X_test = np.hstack([np.ones((X_raw_test.shape[0], 1)), X_raw_test])
        expected_predictions = np.array([0, 1, 2])
        predictions = self.model.predict(X_test)
        np.testing.assert_array_equal(predictions, expected_predictions)

    def test_train_empty_data(self):
        """Test that training with empty data raises an error."""
        X_empty = np.array([])
        Y_empty = np.array([])
        with self.assertRaises(ValueError):
            self.model.train(X_empty, Y_empty)
            
    def test_train_single_example(self):
        """Test that training with a single example does not raise an error."""
        X_single = np.array([[1, 1]]) 
        Y_single = np.array([0])  
        self.model.train(X_single, Y_single)
        self.assertEqual(len(self.model.classifiers), len(self.model.classifiers)) 
        
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
    
    # -----------------
    # Additional Tests
    # -----------------

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
                epochs=100,
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
                epochs=100,
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
                epochs=100,
                random_state=42
            )

    def test_invalid_epochs(self):
        """Test that an invalid `epochs` parameter raises an error."""
        with self.assertRaises(ValueError):
            AllPairsLogisticRegression(
                n_classes=3,
                binary_classifier_class=BinaryLogisticRegression,
                n_features=2,
                batch_size=1,
                epochs=-10,  # Invalid
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
