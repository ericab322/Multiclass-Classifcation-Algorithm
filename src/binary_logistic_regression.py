import numpy as np

class BinaryLogisticRegression:
    def __init__(self, n_features, batch_size, conv_threshold = 1e-4, max_epochs = 100, random_state = None):
        """Initialize the binary logistic regression model.
        @param n_features: Number of features in the dataset, an integer.
        @param batch_size: Batch size for training, an integer.
        @param conv_threshold: Convergence threshold for training, a float.
        @return: None
        """
        if not isinstance(n_features, int) or n_features <= 0:
            raise ValueError("`n_features` must be a positive integer.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("`batch_size` must be a positive integer.")
        if not isinstance(conv_threshold, (int, float)) or conv_threshold <= 0:
            raise ValueError("`conv_threshold` must be a positive number.")
        if not isinstance(max_epochs, int) or max_epochs <= 0:
            raise ValueError("`max_epochs` must be a positive number.")
        if random_state is not None and not isinstance(random_state, int):
            raise ValueError("`random_state` must be an integer or None.")
            
        self.n_features = n_features
        self.weights = np.zeros(n_features + 1)  # extra element for bias
        self.alpha = 0.01
        self.batch_size = batch_size
        self.conv_threshold = conv_threshold
        self.max_epochs = max_epochs
        if random_state is not None:
            np.random.seed(random_state)

    def sigmoid(self, z):
        '''
        Perform sigmoid operation
        @params:
            z: the input to which sigmoid will be applied
        @return:
            an array with sigmoid applied elementwise.
        '''
        return 1 / (1 + np.exp(-z))

    def train(self, X, Y):
        '''self.epochs
        Trains the model using stochastic gradient descent
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            num_epochs: integer representing the number of epochs taken to reach convergence
        '''
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise TypeError("`X` and `Y` must be Numpy arrays.")
        if X.size == 0 or Y.size == 0:
            raise ValueError("`X` and `Y` cannot be empty.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("mismatch in # of samples between `X` and `Y`.")
        if X.shape[1] != self.n_features:
            raise ValueError(f"`X` must have {self.n_features} features.")
        if not np.array_equal(Y, Y.astype(int)) or not np.all((Y == 0) | (Y == 1)):
            raise ValueError("`Y` must contain binary labels (0 or 1).")

        # intializing values
        epochs = 0
        n_examples = X.shape[0]
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])  # Append bias term

        for i in range(self.max_epochs):
            # update # of epochs
            epochs +=1
            # acquire indices for shuffling of X and Y
            indices = np.arange(n_examples)
            np.random.shuffle(indices)
            X_bias = X_bias[indices]
            Y = Y[indices]
            # calc last epoch loss
            last_epoch_loss = self.loss(X, Y)
            # for the # of batches
            for i in range(0, n_examples, self.batch_size):
                X_batch = X_bias[i:i + self.batch_size]
                Y_batch = Y[i:i + self.batch_size]
                # reinitialize gradient to be 0s
                grad = np.zeros(self.weights.shape)
                # for each pair in the batch
                for x, y in zip(X_batch, Y_batch):
                    prediction = self.sigmoid(self.weights @ x) #np.dot(self.weights, x))
                    # gradient calculation
                    error = prediction - y
                    grad += error * x
                # update weights
                self.weights -= ((self.alpha * grad)/ self.batch_size)
            epoch_loss = self.loss(X, Y)
            if abs(epoch_loss - last_epoch_loss) < self.conv_threshold:
                break
        return epochs

    def loss(self, X, Y):
        '''
        Returns the total log loss on some dataset (X, Y), divided by the number of examples.
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding labels for each example
        @return:
            A float number which is the average loss of the model on the dataset
        '''
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise TypeError("`X` and `Y` must be Numpy arrays.")
        if X.size == 0 or Y.size == 0:
            raise ValueError("`X` and `Y` cannot be empty.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("mismatch in # of samples between `X` and `Y`.")
        if X.shape[1] != self.n_features:
            raise ValueError(f"`X` must have {self.n_features} features.")
        if not np.array_equal(Y, Y.astype(int)) or not np.all((Y == 0) | (Y == 1)):
            raise ValueError("`Y` must contain binary labels (0 or 1).")
        
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Append bias term
        n_examples = X.shape[0]
        total_loss = 0

        for i in range(n_examples):
            # linear output (dot product)
            linear_output = X[i] @ self.weights.T  #np.dot(self.weights, X[i].T)
            # calc logistic loss for each sample
            y = 1 if Y[i] == 1 else -1
            logistic_loss = np.log(1 + np.exp(-y * linear_output))
            total_loss += logistic_loss
    
        return total_loss / n_examples
    
    def predict(self, X):
        '''
        Compute predictions based on the learned weigths and examples X
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        if not isinstance(X, np.ndarray):
            raise TypeError("`X` must be a Numpy array.")
        if X.size == 0:
            raise ValueError("`X` cannot be empty.")
        if X.shape[1] != self.n_features:
            raise ValueError(f"`X` must have {self.n_features} features.")
            
        # multiply X by weights of model
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Append bias term
        predictions = self.sigmoid(X @ self.weights.T)
        return np.where(predictions >= 0.5, 1, 0)
        
    def predict_probs(self, X):
        '''
        Compute prediction probabilities based on the learned weigths and examples X
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            an array with sigmoid applied elementwise.
        '''
        if not isinstance(X, np.ndarray):
            raise TypeError("`X` must be a Numpy array.")
        if X.size == 0:
            raise ValueError("`X` cannot be empty.")
        if X.shape[1] != self.n_features:
            raise ValueError(f"`X` must have {self.n_features} features.")
            
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Append bias term
        predictions = self.sigmoid(X @ self.weights.T)
        return predictions

    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise TypeError("`X` and `Y` must be Numpy arrays.")
        if X.size == 0 or Y.size == 0:
            raise ValueError("`X` and `Y` cannot be empty.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("mismatch in # of samples between `X` and `Y`.")
        if X.shape[1] != self.n_features:
            raise ValueError(f"`X` must have {self.n_features} features.")

        predictions = self.predict(X)
        accuracy = np.mean(predictions == Y)
        return accuracy
