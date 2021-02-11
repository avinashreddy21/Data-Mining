import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

train = pd.read_csv('PB1_train.csv', header=None)
test = pd.read_csv('PB1_test.csv', header=None)
train_x = train.iloc[:, :-1]
train_y = train.iloc[:, -1]
test_x = test.iloc[:, :-1]
test_y = test.iloc[:, -1]

# Training a decision Tree using SVM:kernel='linear'
sm1 = svm.SVC(kernel='linear')
sm1.fit(train_x, train_y)
pred = sm1.predict(test_x)
print('Predicted Values on PB1_test.csv using SVM kernel=linear:', pred)
accu=accuracy_score(test_y,pred)
print('Accuracy:', accu*100, '%')

# Training a decision Tree using SVM:kernel='poly'
sm1 = svm.SVC(kernel='poly', degree=5, gamma='scale')
sm1.fit(train_x, train_y)
pred = sm1.predict(test_x)
print('Predicted Values on PB1_test.csv using SVM kernel=poly:', pred)
accu = accuracy_score(test_y, pred)
print('Accuracy:', accu*100, '%')

# Training a decision Tree using SVM:kernel='rbf'
sm1 = svm.SVC(kernel='rbf')
sm1.fit(train_x, train_y)
pred = sm1.predict(test_x)
print('Predicted Values on PB1_test.csv using SVM kernel=rbf:', pred)
accu = accuracy_score(test_y, pred)
print('Accuracy:', accu*100, '%')
