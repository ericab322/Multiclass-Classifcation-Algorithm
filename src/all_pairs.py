import numpy as np
class AllPairsLogisticRegression:
    def __init__(self, n_classes, binary_classifier_class, n_features, batch_size, max_epochs = 100, conv_threshold = 1e-6, random_state=None):
        """
        Initialize the all-pairs logistic regression model approach.
        @param n_classes: Number of classes in the dataset, an integer.
        @param binary_classifier_class: Class for binary logistic regression, a class object.
        @param n_features: Number of features in the dataset, an integer.
        @param batch_size: Batch size for training the binary classifiers, an integer.
        @param conv_threshold: Convergence threshold for training, a float.
        @return: None
        """
        
        if not isinstance(n_classes, int) or n_classes <= 1:
            raise ValueError("`n_classes` must be an integer greater than 1.")
        if not isinstance(max_epochs, int) or max_epochs <= 0:
            raise ValueError("`epochs` must be an integer greater than 0.")
        if not callable(binary_classifier_class):
            raise TypeError("`binary_classifier_class` must be a callable class.")
        if not isinstance(n_features, int) or n_features <= 0:
            raise ValueError("`n_features` must be a positive integer.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("`batch_size` must be a positive integer.")
        if not isinstance(max_epochs, int) or max_epochs <= 0:
                raise ValueError("`max_epochs` must be a positive number.")
        if not isinstance(conv_threshold, (int, float)) or conv_threshold <= 0:
            raise ValueError("`conv_threshold` must be a positive number.")
        
        self.n_classes = n_classes
        self.classifiers = {}
        self.n_features = n_features
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.conv_threshold = conv_threshold
        self.binary_classifier_class = binary_classifier_class
        self.random_state = random_state

    def train(self, X, Y):
        """
        Train the all-pairs logistic regression model by training binary classifiers
        for each pair of classes in the dataset.
        @param X: Input features of the dataset, a numpy array of shape (n_samples, n_features).
        @param Y: Labels of the dataset, a numpy array of shape (n_samples).
        @return: None
        """
        if X.size == 0 or Y.size == 0:
            raise ValueError("Input data `X` and labels `Y` cannot be empty.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Mismatch in number of samples between `X` and `Y`.")
        if np.any((Y < 0) | (Y >= self.n_classes)):
            raise ValueError(f"Labels in `Y` must be in the range [0, {self.n_classes - 1}].")
        unique_classes = np.arange(self.n_classes)
        pairs = [(class_i, class_j) for class_i in unique_classes for class_j in unique_classes if class_i < class_j]

        for class_i, class_j in pairs:
            mask = (Y == class_i) | (Y == class_j)
            SX = X[mask]
            SY = np.where(Y[mask] == class_i, 1, 0)
            classifier = self.binary_classifier_class(
                n_features=self.n_features,
                batch_size=self.batch_size,
                max_epochs=self.max_epochs, random_state=self.random_state,
                conv_threshold = self.conv_threshold
            )
            classifier.train(SX, SY)
            self.classifiers[(class_i, class_j)] = classifier

    def predict(self, X):
        """
        Predict the class labels for the input data using the trained classifiers.
        @param X: Input features to classify, a numpy array of shape (n_samples, n_features).
        @return: Predicted class labels, a numpy array of shape (n_samples).
        """
        if X.size == 0:
            raise ValueError("Input data `X` cannot be empty.")
        if X.shape[1] != self.n_features:
            raise ValueError(f"`X` must have {self.n_features} features.")
        
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
        if X.size == 0 or Y.size == 0:
            raise ValueError("Input data `X` and labels `Y` cannot be empty.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Mismatch in number of samples between `X` and `Y`.")
        
        predictions = self.predict(X)
        correct_predictions = np.sum(predictions == Y)
        return correct_predictions / len(Y)
