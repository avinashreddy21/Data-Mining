import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cosine
np.seterr(divide='ignore', invalid='ignore')


def eucledian(a, b):
    sum = 0
    for i in range(len(a)):
        sum += math.pow(a[i] - b[i], 2)
    return math.sqrt(sum)


def cosine_similarity(a, b):
    return cosine(a, b)


class K_Means:
    def __init__(self, clu=2, con=1e-20, cost = eucledian, no_iter=10000):
        self.clu = clu
        self.no_iter = no_iter
        self.con = con
        self.sse = 0
        self.cost = cost

    def fit(self, d):
        self.centers = {}
        for i in range(self.clu):
            self.centers[i] = d[np.random.choice(len(d), replace=False)]
        for i in range(self.no_iter):
            self.classes = {}
            for i in range(self.clu):
                self.classes[i] = []
            for row in d:
                space = []
                for center in self.centers:
                    space.append(self.cost(self.centers[center], row))
                center_index = space.index(min(space))
                self.classes[center_index].append(row)
            old_centers = dict(self.centers)
            for i in self.classes:
                if len(self.classes[i]) != 0:
                    self.centers[i] = np.mean(self.classes[i], axis=0)
            convergence = True
            for i in self.centers:
                actu_center = old_centers[i]
                pre_center = self.centers[i]
                if(np.abs(np.sum((actu_center - pre_center)/pre_center * 100)) > self.con):
                    convergence = False
            if convergence:
                break
        self.K_index = self.classes
        self.clu_cen = self.centers
        obj_j = 0
        for k in self.centers:
            for d_point in self.classes[k]:
                obj_j += np.square(eucledian(self.centers[k],d_point))
        self.sse = obj_j


def main():
    x = pd.read_csv("overdoses.csv")
    x["Population"] = x["Population"].str.replace(",", "").astype(float)
    x["Deaths"] = x["Deaths"].str.replace(",", "").astype(float)
    a1 = np.array(x["Population"])
    b1 = np.array(x["Deaths"])
    new_table = np.column_stack((a1, b1))
    outputfile1=open('extracting_columns.csv', 'w')
    for i in range(len(new_table)):
        for j in range(len(new_table[0])):
            print(new_table[i][j], file=outputfile1, end=',')
        print(file=outputfile1, end='\n')
    final_matrix = np.zeros((50, 2))
    sse = np.zeros(16)
    for k in range(2, 16):
        model = K_Means(clu=k)
        model.fit(new_table)
        sse[k] = model.sse
        if k == 5:
            n_dict = {}
            for key, values in model.K_index.items():
                for i in values:
                    n_dict[tuple(i)] = key
            for i in range(50):
                final_matrix[i] = [i, n_dict[tuple(new_table[i])]]
    outputfile = open('cluster_number.csv', 'w')
    for i in range(len(final_matrix)):
        for j in range(len(final_matrix[0])):
            print(final_matrix[i][j], file=outputfile, end=',')
        print(file=outputfile, end='\n')
    plt.plot(np.arange(2, 16), sse[2:])
    plt.xlabel('Number of Clusters')
    plt.ylabel('Objective Function Value')
    plt.show()


if __name__ == '__main__':
    main()
