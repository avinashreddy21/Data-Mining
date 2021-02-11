# Clustering from scratch

Accidental deaths by fatal drug overdose is a rising trend in the United States The
overdoses.csv dataset contains information on such opioid related drug overdose fatalities.
It has the 50 rows (one for each state) and the following four columns :
* State : Names of states
* Population : Population in a particular state
* Deaths : Number of opioid casualties in that state
* Abbrev : State abbreviation

K_means_from_scratch.py: To implement K-means clustering algorithm from scratch on the population and deaths columns of dataset.
Run k-means algorithm for a range of values of k (ranging from 2 to 15). For each value of k,
upon convergence (mentioned below) of your k-means algorithm, the objective
function value is computed.
https://medium.com/analytics-vidhya/k-means-clustering-optimizing-cost-function-mathematically-1ccae156299f

K-means convergence criteria :
[1] No further changes in the cluster centers in the next iteration, or
[2] Maximum number of iterations (500) is reached – whichever occurs first.

k_means_cosine_similarity.py: To represent the closeness of
state pairs with respect to their Population and Death values using cosine similarity
metric.

Run the k-means algorithm on this similarity matrix over 50 rows (each row in this new table
represents a data point with 50 features) for a range of values of k, ranging from 2 to 15. For
each value of k, upon convergence, the objective function value is computed.

https://www.machinelearningplus.com/nlp/cosine-similarity/

Convergence criteria :
[1] No further changes in the cluster centers in the next iteration, or
[2] Maximum number of iterations is reached – whichever occurs first.
