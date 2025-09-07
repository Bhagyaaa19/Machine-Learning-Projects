""""
ANN - Basic algorithm

Multiple Perceptron Classifier
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

cancer = load_breast_cancer()

# print(cancer)

# print(cancer['data'].shape)
# (569, 30)
# 569 rows and 30 different variables

# seperate the labels and the input data
# because you cant show the test data to the model

# input has to be a numpy matrix
x = cancer.data  # same thing as cancer['data']
y = cancer.target

# print(x)
# print(y)

# You need to do the split first
X_train, X_test, Y_train, Y_test = train_test_split(x, y)  # order is very important
print(X_train.shape)
print(X_test.shape)

# use this data and calculate the parameters
scaler = StandardScaler().fit(X_train)  # we don't give any information about the test data

X_train = scaler.transform(X_train)
# whatever you do to the training data, you have to do it SEPARATELY to the testing data too
X_test = scaler.transform(X_test)

# Now we build the model
# we build the basic model,
# and we find the best parameters by fine tuning

#                                 3 hidden layers
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=100)
# no need to do any transformation for the y data
mlp.fit(X_train, Y_train)

print(mlp)

fig, ax = plt.subplots(figsize = (6,6))
ax.plot(mlp.loss_curve_) # plotting the trained object mlp
ax.set_xlabel("Number of Iterations")
ax.set_ylabel("Loss")
plt.show()

# test set is only for the evaluation
predictions = mlp.predict(X_test)

print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test, predictions))
