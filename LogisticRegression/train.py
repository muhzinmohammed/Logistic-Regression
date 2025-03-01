import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression


bc = datasets.load_breast_cancer()
X,Y = bc.data,bc.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)

# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:,0],Y,color = "r",marker= "o", s=20)
# plt.show()
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)
pred = classifier.predict(X_test)

def accuracy(y_pred,Y_test):
    return np.sum(y_pred==Y_test)/len(Y_test)

print(accuracy(pred,Y_test))

