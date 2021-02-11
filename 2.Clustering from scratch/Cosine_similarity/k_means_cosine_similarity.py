import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from k_means_from_scratch import K_Means, eucledian


def main():
    x=pd.read_csv("overdoses.csv")
    x["Population"] = x["Population"].str.replace(",", "").astype(float)
    x["Deaths"] = x["Deaths"].str.replace(",", "").astype(float)
    a1 = np.array(x["Population"])
    b1 = np.array(x["Deaths"])
    new_table = np.column_stack((a1, b1))
    sim_mat = np.zeros((50, 50))

    for i in range(len(new_table)):
        a = new_table[i]
        for j in range(len(new_table)):
            sim_mat[i, j] = round(cosine(a, new_table[j]), 20)
    # maxi=np.max(sim_mat)
    # mini=np.min(sim_mat)
    # sim_mat=sim_mat/maxi
    # print(sim_mat.shape)
    outputfile = open('cosine_similarity_matrix.csv', 'w')
    for i in range(len(sim_mat)):
        for j in range(len(sim_mat[0])):
            print(sim_mat[i][j], file=outputfile, end=',')
        print(' ', file=outputfile, end='\n')
    sse = np.zeros(16)
    for k in range(2, 16):
        model = K_Means(k)
        model.fit(sim_mat)
        sse[k] = model.sse
    plt.plot(np.arange(2, 16), sse[2:])
    plt.xlabel('Number of Clusters')
    plt.ylabel('Objective Function Value')
    plt.show()


if __name__ == '__main__':
    main()
