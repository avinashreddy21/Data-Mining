# Linear Regression and Decision Tree Classification models

Linear Regression is performed on “PB1_train.csv” and “PB1_test.csv” and “PB2_train.csv” and “PB2_test.csv”

Datasets “PB1_train.csv” and “PB1_test.csv” have three columns first two columns are the features
x =[x<sub>1</sub> ,x<sub>2</sub> ]) and the last column is the prediction value (y). 

The hypotheses formula is given
y = θ<sub>0</sub> + θ<sub>1</sub> x<sub>1</sub> + θ<sub>2</sub> x<sub>2</sub>

Trained a linear regression model,
M, over x and y values from “PB1_train.csv” and reported the
corresponding model parameters ( θ<sub>0</sub> , θ<sub>1</sub> and θ<sub>2</sub> ).

Tested model on “PB1_test.csvPB1_test.csv” and reported the predicted values (values (y̅)) for each row. Calculated the mean squared error between the predicted values and original values (third column in “PB1_test.csv”).

Decision Tree Classification is performed on “PB3_train.csv” and “PB3_test.csv” and “PB4_train.csv” and “PB4_test.csv”

In this classification problem, you are required to train a Decision tree model that predicts whether a
person is male represented as 0 or female represented a s 1 given three features : height (in centimeters), age and weight (in kilograms)

Used “PB3_train.csv” and “PB3_test.csv” and “PB4_train.csv” and “PB4_test.csv” for this classification, where the first three columns represent three
features (height, age, weight), and the fourth column represent class label (0/1). 

Trained a decision tree DT (using Gini index metric) on “PB3_train.csv” and “PB4_train.csv” data that learns to map the mentioned features to their
corresponding class values.

Gini Index: https://medium.com/analytics-steps/understanding-the-gini-index-and-information-gain-in-decision-trees-ab4720518ba8

Reported the predicted values y̅ and accuracy percentage (percentage of matches) of the model DT by
testing it on “PB3_test.csv” and “PB4_test.csv” data.
