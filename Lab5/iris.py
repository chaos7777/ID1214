import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
#print (iris.feature_names)
#print (iris.target_names)
#for i in range (len(iris.target)):
#        print( "Example %d: Label %s featrues: %s" % (i , iris.target[i], iris.data[i]))

idx = [0 , 50 , 100]
train_target = np.delete(iris.target , idx)
train_data = np.delete(iris.data , idx , axis=0)

test_target = iris.target[idx]
test_data = iris.data[idx]

clf = tree.DecisionTreeClassifier()
cld = clf.fit(train_data , train_target)

print (test_target)
print (clf.predict(test_data))
