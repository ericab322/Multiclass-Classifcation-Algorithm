import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.all_pairs import AllPairsLogisticRegression
from src.binary_logistic_regression import BinaryLogisticRegression

df = pd.read_csv("/Users/ericabrown/data2060/Multiclass-Classification-Algorithm/data/processed/obesity_standardized.csv")
X = df.drop("NObeyesdad", axis=1).values  
Y = df["NObeyesdad"].values 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

n_classes = len(np.unique(Y))  
n_features = X_train.shape[1]
batch_size = 10
epochs = 100
conv_threshold=0.01
binary_classifier_class = BinaryLogisticRegression

model = AllPairsLogisticRegression(n_classes=n_classes,
                                   binary_classifier_class=binary_classifier_class,
                                   n_features=n_features, 
                                   batch_size=batch_size,
                                   conv_threshold=conv_threshold, 
                                   epochs=epochs
                                )
model.train(X_train, Y_train)
accuracy = model.accuracy(X_test, Y_test)
print(f"All-Pairs Logistic Regression Accuracy: {accuracy:.4f}")
