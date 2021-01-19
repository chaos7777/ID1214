from sklearn import tree
#Labels 0 = BMW,  1 = Audi,  2 = Mercedes,  Chrysler = 3
#Feature 1 Engine HP
#Feature 2 Number of cylinders
#Feature 3 Transmission Type | 0 = Manual,  1 = Automatic
#Feature 4 Driven wheels | 0 = Rear wheel drive,  1 = All wheel drive,  2 = Front wheel drive
#Feature 5 Number of doors | 2 = 0,  4 = 1
#Feature 6 Category | 0 = Performance, 1 = Luxury, 2 = Flex fuel

features = [[335,6,0,0,0,0],[300,6,0,0,0,0],[230,6,0,0,0,1],[300,6,0,0,0,1],[320,6,0,0,0,0],[248,4,1,0,0,0],[172,6,0,2,1,1],[172,6,1,1,1,1],[172,6,0,1,1,1],[172,6,1,2,1,1],[130,4,0,0,1,1],[158,6,0,0,1,1],[177,6,1,0,1,1],[177,6,1,1,1,1],[228,6,0,0,0,0],[217,6,1,0,0,1],[184,4,1,2,1,2],[295,6,1,1,1,2],[185,4,1,2,1,2],[183,4,1,2,1,2],[300,6,1,0,1,0]]
labels = [0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)

#Expected value is [2] : Mercedes
print(clf.predict([[177,6,1,1,1,1]]))

#Expected value is [3] : Chrysler
print(clf.predict([[292,6,1,0,1,0]]))

#Expected value is [0] : BMW
print(clf.predict([[240,4,0,0,0,0]]))
