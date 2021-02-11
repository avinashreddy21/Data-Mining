import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv('PB4_train.csv', header=None)
test = pd.read_csv('PB4_test.csv', header=None)
train_x = train.iloc[:, :-1]
train_y = train.iloc[:, -1]
test_x = test.iloc[:, :-1]
test_y = test.iloc[:, -1]


def trg(xtrain, ytrain):
    # Training a decision Tree using Gini-index Metric
    fittin = DecisionTreeClassifier(criterion='gini')
    fittin.fit(xtrain, ytrain)
    return fittin


def predic(xtest, fit):
    pred = fit.predict(xtest)
    return pred


def accuracy(actual, predicted):
    return accuracy_score(actual, predicted)


trained = trg(train_x, train_y)
predicted = predic(test_x, trained)
print('Predicted Values on PB4_test.csv:', predicted)
accu = accuracy(test_y, predicted)
print('Accuracy:', round(accu*100, 2), '%')
