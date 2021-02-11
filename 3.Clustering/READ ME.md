# Different Clustering techniques

* Algorithm (a) : K-means (Initialize k cluster centers by randomly picking up k points among all data
points in the dataset).
* Algorithm (b) : A clustering technique of distributing all the data-points in the dataset into k groups
(clusters) such that diameter of the largest cluster is minimum among all possible ways of creating k
clusters out of these data-points (Diameter = Euclidean distance between the two farthest points in a
cluster).
* Algorithm (c) : Spectral-Clustering (Use a Gaussian kernel for computing affinity score between two
points. Use k-nearest neighbor for graph construction (set k=5). You may use libraries for sub-tasks in
spectral-clustering, for example- computing diagonal Degree matrix, Eigen-vectors & Eigen-values).

The 3 algorithms are implemented on 3 different datasets to understand which one is better.

From the results, 
* Algorithm (a) gave the worst performance.
* In algorithm (b), two farthest points (for k=2) are chosen as cluster centers and the remaining datapoints are assigned to the center which is closer to them. Each time, the next center is chosen as far as possible from the centers that are chosen up to previous iteration. Consider a point A(x,y) in any cluster which is farthest to all the centers chosen and distance between A and its nearest center be ‘R’, then every point in that cluster must be within the distance of at most ‘2R’ (diameter of cluster).
This kind of prediction cannot be done using algorithm (a).
* Moreover, the error rate in algorithm (b) is 100% where as the error rate in algorithm (a) is greater than 100%.
* Algorithm (c) gave the best performance.
* K-means, as a clustering algorithm, is ideal for discovering clusters like the ones where all members of each cluster are closer to each other whereas in spectral clustering, datapoints are not clustered directly, instead a similarity matrix is derived from data points.
* The clustering in algorithm (c) is performed on Eigen vectors of matrices that are derived from the given dataset rather than performing clustering on dataset directly, thus improving the cluster analysis. The data is represented in low-dimensional space (graphical representation) which can be clustered easily and accurately.
