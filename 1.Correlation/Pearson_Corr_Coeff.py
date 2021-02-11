import pandas as pd
from statistics import mean
import math

data = pd.read_csv('overdoses.csv')
data.Population = data.Population.str.replace(",", "")
data.Deaths = data.Deaths.str.replace(",", "")

a = list(data.Population)
b = list(data.Deaths)

float_a = [float(i) for i in a]
float_b = [float(j) for j in b]

mean_a = mean(float_a)
mean_b = mean(float_b)

assert len(float_a) == len(float_b)
length = len(float_a)
abproduct = 0
a_sqdiff = 0
b_sqdiff = 0

for k in range(length):
    a_diff = float_a[k] - mean_a
    b_diff = float_b[k] - mean_b
    abproduct += (a_diff * b_diff)
    a_sqdiff += (a_diff * a_diff)
    b_sqdiff += (b_diff * b_diff)

print("Pearson correlation coefficient between population and deaths: ", abproduct/math.sqrt(a_sqdiff * b_sqdiff))
