import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('overdoses.csv')
data.Population = data.Population.str.replace(",", "")
data.Deaths = data.Deaths.str.replace(",", "")

a = list(data.Population)
b = list(data.Deaths)
c = list(data.Abbrev)

population_a = [float(i) for i in a]
deaths_b = [float(j) for j in b]
staab_c = [str(k) for k in c]

assert len(population_a) == len(deaths_b)
length = len(population_a)
odd = []

for id in range(length):
    odd.append(deaths_b[id]/population_a[id])
# print(odd)
plt.bar(staab_c, odd)
plt.xlabel('Abbreviations of State', fontsize=10)
plt.ylabel('Opioid Death Density(ODD)', fontsize=10)
plt.title('Bar-Graph representing ODD')
plt.show()
