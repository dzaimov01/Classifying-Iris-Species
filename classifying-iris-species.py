import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
print("Keys of iris_dataset: {}\n.".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: {}".format(iris_dataset['feature_names']))
print("Data: {}".format(iris_dataset['data'].shape))
print("First 5 columns of data: \n{}".format(iris_dataset['data'][:5]))
print("Type of targets {}: ".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("Target: \n{}".format(iris_dataset['target']))

X = np.array(iris_dataset['data'])
y = np.array(iris_dataset['target'])

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25)

''' 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
'''

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

#Show the data
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
acc = knn.score(X_test, y_test)
print(acc)

print("Enter sepal length: ")
sepal_length = input()
print("Enter sepal width: ")
sepal_width = input()
print("Enter petal length: ")
petal_length = input()
print("Enter petal width: ")
petal_width = input()

prediction_array = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = knn.predict(prediction_array)
predicted_name = ""
if prediction == 0:
    predicted_name = "setosa"
elif prediction == 1:
    predicted_name = "versicolor"
elif prediction == 2:
    predicted_name = "virginica"

print("This plant is from type", predicted_name)

