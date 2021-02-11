# Correlation

Accidental deaths by fatal drug overdose is a rising trend in the United States The
overdoses.csv dataset contains information on such opioid related drug overdose fatalities.
It has the 50 rows (one for each state) and the following four columns :
* State : Names of states
* Population : Population in a particular state
* Deaths : Number of opioid casualties in that state
* Abbrev : State abbreviation

Correlation is any statistical association that refers to how close two variables are to
having a linear relationship with each other. Pearson correlation coefficient is such a measure
of the linear correlation between two variables X and Y.

Pearson_Corr_Coeff.py: To calculate the Pearson correlation coefficient between the Population and
Deaths columns using Python libraries.

Optoid_Death_Density.py: To generate bar-graph representing the Opioid Death Density (ODD), Opioid
Death Density = Number of deaths in the state/Population for that state , for each state.

Similarity_Matrix.py: To generate a similarity matrix representing the closeness of state pairs with respect to
their ODD- a state pair will have a similarity value of 1 if the difference in their ODD values is
0, and will have a value of 0 if difference in their ODD values is maximum among the ODD
values of all the given pairs.
