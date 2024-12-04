class OneVsAllLogisticRegression:
    def __init__(self, n_classes, binary_classifier_class, n_features, batch_size, conv_threshold):
        """
        Initialize the One-vs-All logistic regression model.
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
        self.binary_classifier_class = binary_classifier_class
        self.conv_threshold = conv_threshold

    def train(self, X, Y):
        """
        Train the One-vs-All logistic regression model by training one binary classifier
        for each class in the dataset.
        @param X: Input features of the dataset, a numpy array of shape (n_samples, n_features).
        @param Y: Labels of the dataset, a numpy array of shape (n_samples,).
        @return: None
        """
        for class_i in range(self.n_classes):
            # Create binary labels: 1 for the current class, 0 for others
            binary_labels = np.where(Y == class_i, 1, 0)
            classifier = self.binary_classifier_class(
                n_features=self.n_features,
                batch_size=self.batch_size,
                conv_threshold=self.conv_threshold
            )
            classifier.train(X, binary_labels)
            self.classifiers[class_i] = classifier

    def predict(self, X):
        """
        Predict the class labels for the input data using the trained classifiers.
        @param X: Input features to classify, a numpy array of shape (n_samples, n_features).
        @return: Predicted class labels, a numpy array of shape (n_samples,).
        """
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
        preds = self.predict(X)
        acc = np.mean(preds == Y)
        return acc