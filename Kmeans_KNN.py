# simple Python demonstration of clustering and nearest neighbors classification using scikit-learn
# 
# 
# 
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

data = {
    'x': [25, 34, 22, 27, 33, 33, 31, 22, 35, 34, 67, 54, 57, 43, 50, 57, 59, 52, 65, 47, 49, 48, 35, 44, 45, 43, 51,
          46],
    'y': [79, 51, 53, 78, 59, 74, 73, 57, 69, 75, 51, 32, 40, 47, 53, 36, 35, 58, 59, 50, 25, 20, 14, 12, 20, 27, 8, 7]
}

df = DataFrame(data, columns=['x', 'y'])
# print(df)
# Learn this code the hard way alone

# model fitting - fitting the data to the model
km = KMeans(n_clusters=4).fit(df)

# centroids are the x and y mean values of the threw clusters
centroids = km.cluster_centers_

print(centroids)
# printing the labels of the clusters
# the labels means different indexes - we match these to color the data points
print('labels', km.labels_)

# new data set to predict ( better to use np values
xnew = np.array([[26, 70], [33, 40], [25, 50], [40, 30]])
print(xnew)
predictedValues = km.predict(xnew)
print('Predicted values:', predictedValues)

# plotting the input data
# assigning colors to the different data
plt.scatter(df['x'], df['y'], c=km.labels_.astype(float), s=50, alpha=0.5)  # float for continuous mapping of colors
# s = size, alpha = making the data point transparent with 1 being the most opaque

# labelling a scatter plot one by one - hence we use a for loop
# i is index and lbl is the label
# when using enumerate in a list - we get the index and the value both, but for a for loop we get only  the value
for i, lbl in enumerate(km.labels_):
    plt.annotate(lbl, (df['x'][i], df['y'][i]))
    # we give the x index and the y index of eachdata point and assign the relavant label
# marking the centroid on the same plot
# centroids is a matrix
# plt.scatter(centroids[x], centroids[y])
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)  # give all the rows in the first column

# plotting the predicted values
plt.scatter(xnew[:, 0], xnew[:, 1], c='green', s=50)
# annotating the predicted values
for i, lbl in enumerate(predictedValues):
    plt.annotate(lbl, (xnew[:, 0][i], xnew[:, 1][i]))

plt.show()  # we can add several plots to the same figure if thees only 1 plt.show(), to plot two different ones, we need subplots check :)
# you can get false classifications
# we have to play with the k values then


# KNN =======================================================================================
# importing a dataset from the package
iris = datasets.load_iris()
print(iris)
# extracting the labels
labels = iris.target
print("Labels are")
print(labels)

features = iris.data  # the name given by the dataset
print("Data are")
print(features)

# fit the standardized data from the features matrix
scaler = StandardScaler().fit(features)
print(scaler)  # in here there is the model

# use the model to transfrom the data
stdata = scaler.transform(features)  # the data has a miu of 0 and st dev of 1 one now - normalized
# data is not transformed

# test data
test_data = [[5.9, 0.5, 1.0, 0.2]]  # don't forget to transform the test data
# whatever you do to the train data has to be done to the test data
testdata_st = scaler.transform(test_data)
print(testdata_st)

sepal_data = stdata[:, [0, 1]]
sepal_testdata = testdata_st[:, [0, 1]]  # always use the transformed data

# finding the two closest neighbors
nn = NearestNeighbors(n_neighbors=2).fit(sepal_data)  # use the final final data set here

distance, indices = nn.kneighbors(sepal_testdata)  # gives the distance and the indexes of the two nearest neighbors

print("Indices are:", indices)
print("Distance", distance)
# take the vote and make the prediction alone

print(labels[indices])

