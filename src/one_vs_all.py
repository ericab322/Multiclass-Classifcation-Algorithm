import numpy as np

class OneVsAllLogisticRegression:
    def __init__(self, n_classes, binary_classifier_class, n_features, batch_size, epochs, random_state = None):
        """
        Initialize the One-vs-All logistic regression model.
        @param n_classes: Number of classes in the dataset, an integer.
        @param binary_classifier_class: Class for binary logistic regression, a class object.
        @param n_features: Number of features in the dataset, an integer.
        @param batch_size: Batch size for training the binary classifiers, an integer.
        @param conv_threshold: Convergence threshold for training, a float.
        @return: None
        """

        if not isinstance(n_classes, int) or n_classes <= 1:
            raise ValueError("`n_classes` must be an integer greater than 1.")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("`epochs` must be an integer greater than 0.")
        if not callable(binary_classifier_class):
            raise TypeError("`binary_classifier_class` must be a callable class.")
        if not isinstance(n_features, int) or n_features <= 0:
            raise ValueError("`n_features` must be a positive integer.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("`batch_size` must be a positive integer.")
      
        self.n_classes = n_classes
        self.classifiers = {}  
        self.n_features = n_features
        self.batch_size = batch_size
        self.epochs = epochs 
        self.binary_classifier_class = binary_classifier_class
        self.random_state = random_state
        

    def train(self, X, Y):
        """
        Train the One-vs-All logistic regression model by training one binary classifier
        for each class in the dataset.
        @param X: Input features of the dataset, a numpy array of shape (n_samples, n_features).
        @param Y: Labels of the dataset, a numpy array of shape (n_samples,).
        @return: None
        """

        if X.size == 0 or Y.size == 0:
            raise ValueError("Input data `X` and labels `Y` cannot be empty.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Mismatch in number of samples between `X` and `Y`.")
        if np.any((Y < 0) | (Y >= self.n_classes)):
            raise ValueError(f"Labels in `Y` must be in the range [0, {self.n_classes - 1}].")
        
        for class_i in range(self.n_classes):
            # Create binary labels: 1 for the current class, 0 for others
            binary_labels = np.where(Y == class_i, 1, 0)
            classifier = self.binary_classifier_class(
                n_features=self.n_features,
                batch_size=self.batch_size,
                epochs=self.epochs, random_state=self.random_state
            )
            classifier.train(X, binary_labels)
            self.classifiers[class_i] = classifier

    def predict(self, X):
        """
        Predict the class labels for the input data using the trained classifiers.
        @param X: Input features to classify, a numpy array of shape (n_samples, n_features).
        @return: Predicted class labels, a numpy array of shape (n_samples,).
        """

        if X.size == 0:
            raise ValueError("Input data `X` cannot be empty.")
        if X.shape[1] != self.n_features:
            raise ValueError(f"`X` must have {self.n_features} features.")


        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.n_classes))

        for class_i, classifier in self.classifiers.items():
            # Get probabilities for the current class
            scores[:, class_i] = classifier.predict_probs(X)

        # Select the class with the highest probability/score for each sample
        return np.argmax(scores, axis=1)

    def accuracy(self, X, Y):
        """
        Calculate the accuracy of the model on the input data and labels.
        @param X: Input features of the dataset, a numpy array of shape (n_samples, n_features).
        @param Y: True labels of the dataset, a numpy array of shape (n_samples,).
        @return: Accuracy of the model as a float between 0 and 1.
        """
        if X.size == 0 or Y.size == 0:
            raise ValueError("Input data `X` and labels `Y` cannot be empty.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Mismatch in number of samples between `X` and `Y`.")
        
        preds = self.predict(X)
        acc = np.mean(preds == Y)
        return acc