import numpy as np
class AllPairsLogisticRegression:
    def __init__(self, n_classes, binary_classifier_class, n_features, batch_size, epochs):
        """
        Initialize the all-pairs logistic regression model approach.
        @param n_classes: Number of classes in the dataset, an integer.
        @param binary_classifier_class: Class for binary logistic regression, a class object.
        @param n_features: Number of features in the dataset, an integer.
        @param batch_size: Batch size for training the binary classifiers, an integer.
        @param conv_threshold: Convergence threshold for training, a float.
        @return: None
        """
        self.n_classes = n_classes
        self.classifiers = {} 
        self.n_features = n_features
        self.batch_size = batch_size
        self.epochs = epochs
        self.binary_classifier_class = binary_classifier_class

    def train(self, X, Y):
        """
        Train the all-pairs logistic regression model by training binary classifiers
        for each pair of classes in the dataset.
        @param X: Input features of the dataset, a numpy array of shape (n_samples, n_features).
        @param Y: Labels of the dataset, a numpy array of shape (n_samples).
        @return: None
        """
        unique_classes = np.arange(self.n_classes)
        pairs = [(class_i, class_j) for class_i in unique_classes for class_j in unique_classes if class_i < class_j]

        for class_i, class_j in pairs:
            mask = (Y == class_i) | (Y == class_j)
            SX = X[mask]
            SY = np.where(Y[mask] == class_i, 1, -1)
            classifier = self.binary_classifier_class(
                    n_features=self.n_features,
                    batch_size=self.batch_size,
                    epochs=self.epochs
                )
            classifier.train(SX, SY)
            self.classifiers[(class_i, class_j)] = classifier

    def predict(self, X):
        """
        Predict the class labels for the input data using the trained classifiers.
        @param X: Input features to classify, a numpy array of shape (n_samples, n_features).
        @return: Predicted class labels, a numpy array of shape (n_samples).
        """
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, self.n_classes), dtype=int)
        for (class_i, class_j), classifier in self.classifiers.items():
            predictions = classifier.predict(X)
            votes[:, class_i] += (predictions == 1)
            votes[:, class_j] += (predictions == 0)
        return np.argmax(votes, axis=1)

    def accuracy(self, X, Y):
        """
        Calculate the accuracy of the model on the input data and labels by finding ratio of correct predictions to total samples.
        @param X: Input features of the dataset, a numpy array of shape (n_samples, n_features).
        @param Y: True labels of the dataset, a numpy array of shape (n_samples).
        @return: Accuracy of the model as a float between 0 and 1.
        """
        predictions = self.predict(X)
        correct_predictions = np.sum(predictions == Y)
        return correct_predictions / len(Y)