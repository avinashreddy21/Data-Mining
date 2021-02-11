import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def rms(real, pre):
    e = 0
    for i in range(len(real)):
        e += ((pre[i]-real[i])**2)
    return e/(len(pre))


train = pd.read_csv('PB1_train.csv', header=None)
test = pd.read_csv('PB1_test.csv', header=None)
train_x = train.iloc[:, 0:2]
train_y = train.iloc[:, -1]
test_x=test.iloc[:, 0:2]
test_y=test.iloc[:, -1]
model = LinearRegression()
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
pred_y1 = pred_y.reshape(30, 1)
coef = model.coef_
intercept = model.intercept_
error = rms(test_y, pred_y)

print('Theta intercept:', intercept)
print('Theta1 and Theta2 Parameters:', coef)
print('Predicted values(y) on PB1_test.csv :', pred_y)
print('Mean square error on the test set:', error)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
xs = test[0]
ys = test[1]
zs = test[2]

ax.scatter(xs, ys, zs, c='red', alpha=0.6, edgecolors='w')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('3-D Plot of PB1_test.csv')
x_surf, y_surf = np.meshgrid(np.linspace(test[0].min(), test[0].max(), 30), np.linspace(test[1].min(), test[1].max(), 30))
z_surf = np.linspace(pred_y.min(), pred_y.max(), 30)
ax.plot_wireframe(x_surf, y_surf, z_surf.reshape(-1, 1), color='blue', alpha=0.6)
plt.show()
