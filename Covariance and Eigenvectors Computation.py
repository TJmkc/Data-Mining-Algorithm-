import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import sys


data = pd.read_csv(sys.argv[1])
data2 = data.drop(['date', 'rv2'], axis=1)




D = data2.values
row_mean = np.array(round(np.mean(data2, axis=0), 2))
print(row_mean)
sum1 = 0
for i in range(len(D)):
    a1 = (D[i] - row_mean)
    sum1 += (np.linalg.norm(a1))**2

var = (1/len(D)) * sum1
print(var)

n = len(D)


dbar = D - row_mean
cov1_inner = ( 1 /len(D)) * np.dot(dbar.T, dbar)
print(cov1_inner)

sum2 = 0
for i in range(len(D)):
    b1 = dbar[i].reshape(1, len(dbar[i]))
    b2 = dbar[i].reshape(len(dbar[i]), 1)
    sum2 += np.dot(b2, b1)
cov2_outer = ( 1 /len(D)) * sum2
print(cov2_outer)

def magnitude(a):
    sum1 = 0
    for i in range(len(a)):
        sum1 += pow(a[i], 2)
    magnitude = sqrt(sum1)
    return magnitude


corr_dict1 = {}
for i in range(len(data2.columns) - 1):
    for m in range(i + 1, len(data2.columns)):
        j = data2.columns[i]
        n = data2.columns[m]
        attr1 = magnitude(dbar[:, i])
        attr2 = magnitude(dbar[:, m])
        corr = ((dbar[:, i]) / attr1).T @ ((dbar[:, m]) / attr2)
        a = {j + ' and ' + n: corr}
        corr_dict1.update(a)

corr_list = []
for i in range(len(data2.columns)):
    for j in range(len(data2.columns)):
        attr1 = float(np.dot(np.mat(dbar[:, i],), np.mat(dbar[:, j]).T))
        attr2 = magnitude(dbar[:, i]) * magnitude(dbar[:, j])
        cos = attr1 / attr2
        corr_list.append(cos)
matrix0 = np.array(corr_list)
corr_matrix = matrix0.reshape((len(data2.columns),len(data2.columns)))
print('The correlation matrix is {}'.format(corr_matrix))


print('The most anti-correlated is %f',sorted(corr_dict1.items(), key = lambda x: x[1], reverse= False)[0])
print('The most correlated is %f',sorted(corr_dict1.items(), key= lambda x: x[1], reverse= True)[0])


dict2 = {}
for key, values in corr_dict1.items():
    if values >= 0:
        b = {key: values}
        dict2.update(b)
print('The most least correlated is %f', sorted(dict2.items(), key=lambda x: x[1], reverse=False)[0])


plt.figure(figsize=(20, 12))
plt.scatter(data2['T6'], data2['T_out'],  alpha=0.5)
plt.xlabel('T6')
plt.ylabel('T_out')
plt.show()
plt.savefig('Most Correlated Pairs')



plt.figure(figsize=(20, 12))
plt.scatter(data2['RH_6'], data2['T7'], alpha=0.5)
plt.xlabel('RH_6')
plt.ylabel('T7')
plt.show()
plt.savefig('Most Anti-Correlated Pairs')



plt.figure(figsize=(20, 12))
plt.scatter(data2['Visibility'], data2['Appliances'], alpha=0.5)
plt.xlabel('Visibility')
plt.ylabel('Appliances')
plt.show()
plt.savefig('Least Correlated Pairs')



x1 = np.random.rand(27, 2)
unit_x1 = x1 / np.linalg.norm(x1, ord=1)


def poweriteration2d(x, cov, t):
    new_x = (cov @ x)
    a = new_x[:, 0]
    b = new_x[:, 1]

    b = b - ((b @ a) / (a @ a)) * a
    lamda_a = max(abs(a))
    lamda_b = max(abs(b))

    a = a / lamda_a
    b = b / lamda_b

    a = a / np.linalg.norm(a, ord=1)
    b = b / np.linalg.norm(b, ord=1)

    x1 = np.array([a, b]).T
    diff = np.linalg.norm(x1 - x)
    if diff <= t:
        return a, lamda_a, b, lamda_b, x1
    return poweriteration2d(x1, cov, sys.argv[2])
eigenvector1, eigenvalue1, eigenvector2, eigenvalue2, x1 = poweriteration2d(unit_x1, cov1_inner, sys.argv[2])

print('The first eigenvector is {} and the corresponding eigenvalues is {}'.format(eigenvector1, eigenvalue1))
print('The second eigenvector is {} and the corresponding eigenvalues is {}'.format(eigenvector2, eigenvalue2))






projection = (dbar @ x1)

plt.figure(figsize=(20,12))
plt.scatter(projection[:, 0], projection[:, 1], alpha=0.5)

plt.xlabel('X')
plt.ylabel('Y')

plt.show()
plt.savefig('Projections on two dimension')
