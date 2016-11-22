import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

trainX = [[1,0,1,1],[0,1,1,0],[1,0,0,1],[0,0,1,0],[0,1,0,1],[0,0,0,0],[1,0,1,1],[0,0,1,1],[1,0,0,0],[0,0,0,1]]
trainY = [1,0,1,1,0,1,1,0,1,0]
testX = [[0,0,1,1],[0,0,1,0],[0,1,0,0]]
trainX = np.asarray(trainX)
trainY = np.asarray(trainY)
testX = np.asarray(testX)

for i in range(2,6):
	kf = KFold(n_splits=5)
	total_error = 0
	for train_index, test_index in kf.split(trainX):
		clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(i,))
		clf.fit(trainX[train_index],trainY[train_index])
		total_error += 2 - np.sum(trainY[test_index] == clf.predict(trainX[test_index]))
	print("With " + str(i) + " hidden units, cross validation error is " + str(total_error / 5.0))

clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(5,))
clf.fit(trainX, trainY)
print(clf.predict(testX))