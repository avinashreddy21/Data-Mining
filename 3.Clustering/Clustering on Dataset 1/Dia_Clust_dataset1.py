import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cosine
np.seterr(divide='ignore', invalid='ignore')


# Calculating Euclidean Distance
def eucledian(a, b):
    sum = 0
    for i in range(len(a)):
        sum += math.pow(a[i] - b[i], 2)
    return math.sqrt(sum)


# Calculating Cosine Similarity
def cosine_similarity(a, b):
    return cosine(a, b)


class K_Means:
    def __init__(self, clu=2, con=1e-20, cost=eucledian, no_iter=1000):
        self.clu = clu
        self.no_iter = no_iter
        self.con = con
        self.sse = 0
        self.cost = cost

    def fit(self, d):
        self.centers = {}
        self.centers[0] = d[np.random.choice(len(d), replace=False)]
        for i in range(1, self.clu):
            mai = []
            for row in d:
                dist = []
                for center in self.centers:
                    dist.append(self.cost(self.centers[center], row))
                maxi = max(dist)
                mai.append(maxi)
            max_index = mai.index(max(mai))
            self.centers[i] = d[max_index]
        for i in range(self.no_iter):
            self.classes = {}
            for i in range(self.clu):
                self.classes[i] = []
            for row in d:
                space = []
                for center in self.centers:
                    # cost is a parameter which is given to find Cosine or Euclidean
                    space.append(self.cost(self.centers[center], row))
                center_index = space.index(min(space))
                self.classes[center_index].append(row)
            old_centers = dict(self.centers)
            for i in self.classes:
                if len(self.classes[i]) != 0:
                    self.centers[i] = np.mean(self.classes[i], axis=0)  # calculating mean of all the points to form the new centres
            convergence = True
            for i in self.centers:
                actu_center = old_centers[i]
                pre_center = self.centers[i]
                if(np.abs(np.sum(((actu_center - pre_center)/pre_center * 100))) > self.con):
                    convergence = False
            if convergence:
                break
        self.K_index = self.classes
        self.clu_cen = self.centers


colors = 10*["r", "b", "c"]
coln = ['f1', 'f2', 'Y']
x = pd.read_csv('dataset1.csv', names=coln)
cl0 = x.loc[x['Y'] == 0.0]
cl1 = x.loc[x['Y'] == 1.0]
cl2 = x.loc[x['Y'] == 2.0]
cla0 = cl0.values[:, :-1]
cla1 = cl1.values[:, :-1]
cla2 = cl2.values[:, :-1]
plt.scatter(cla0[:, 0], cla0[:, 1], color='red')
plt.scatter(cla1[:, 0], cla1[:, 1], color='blue')
plt.scatter(cla2[:, 0], cla2[:, 1], color='black')
plt.title("Input Data")
plt.show()
xy = x.values
clutab = xy[:, :-1]
model = K_Means(clu=2)
model.fit(xy)
for center in model.classes:
    color = colors[center]
    for point in model.classes[center]:
        plt.scatter(point[0], point[1], marker="x", color=color, s=150, linewidths=5)
    for center in model.centers:
        plt.scatter(model.centers[center][0], model.centers[center][1], marker="o", color="k", s=150, linewidths=5)
plt.title("Clustering results using Diameter clustering")
plt.show()
