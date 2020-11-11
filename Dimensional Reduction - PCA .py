
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

data = pd.read_csv(sys.argv[1])
data = data.drop(columns=['date', 'rv2'])
D = data.values
total_var = np.var(D.transpose())
mean_vector = np.array(round(np.mean(data, axis=0), 2))
sum1 = 0
for i in range(len(D)):
    a = (D[i] - mean_vector)
    sum1 += (np.linalg.norm(a))**2
var = (1/len(D)) * sum1

dbar = D - mean_vector
cov = np.cov(dbar, rowvar=False, bias=True)
eigenvalues, eigenvectors = np.linalg.eigh(cov)

def top_pcs(vectors, values, alpha):
    a = 0
    alpha = float(sys.argv[2])
    for i in range(len(values)-1, -1, -1):
        a = a + values[i]
        if (a/var) >= alpha:
            return i, eigenvectors[-i: ]

top_values, top_vectors = top_pcs(eigenvectors, eigenvalues, sys.argv[2])
print('The dimensions that are required capture α=0.975 fraction of the total variance are {}'.format(len(eigenvalues)-top_values))

top_three_vectors = eigenvectors[-3:]
mse = 0
mse = var - eigenvalues[24:27].sum()
print('The MSE is {}'.format(mse))

projection = np.dot(dbar, top_three_vectors.T)

ax = plt.figure(figsize=(14, 10)).add_subplot(111, projection='3d')
xs = projection[:, 0]
ys = projection[:, 1]
zs = projection[:, 2]
ax.scatter(xs, ys, zs, c='r', s=1)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()



