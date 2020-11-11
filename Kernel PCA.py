import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
data = pd.read_csv(sys.argv[1])

data = data.drop(columns = ['date','rv2'])

subdata = data[:5000]

D = subdata.values

K = np.dot(D,D.T)

m1 = (np.identity(5000) - ((1/5000)*np.ones([5000,5000])))

kbar = np.dot(np.dot(m1,K),m1)

eigenvalues, eigenvectors = np.linalg.eigh(kbar)


def variance(eigenvalues, eigenvectors):
    variance_list = []
    total_var = 0
    for i in range(len(eigenvalues)):
        variance_list.append(eigenvalues[i]/5000)
    for i in range(len(variance_list)):
        if variance_list[i]<=0:
            continue
        else:
            total_var += variance_list[i]
    for i in range(len(eigenvalues)):
        if eigenvalues[i]<=0:
            continue
        else:
            eigenvectors[:,i] = (math.sqrt(1/eigenvalues[i]))*eigenvectors[:,i]
    return variance_list,total_var,eigenvectors

variance_list = []
for i in range(len(eigenvalues)):
    variance_list.append(eigenvalues[i]/5000)

total_var = 0
for i in range(len(variance_list)):
    if variance_list[i] <0:
        continue
    else:
        total_var += variance_list[i]

print(total_var)

for i in range(len(eigenvalues)):
    if eigenvalues[i]<0:
        continue
    else:
        eigenvectors[:,i] = (math.sqrt(1/eigenvalues[i]))*eigenvectors[:,i]

def top_pcs(vectors, variance, var ,alpha):
    a = 0
    alpha = float(sys.argv[2])
    for i in range(len(variance)-1,-1,-1):
        a = a + variance[i]
        if (a/var) >= alpha:
            return i,vectors[:,-i:]

top_values, top_vectors = top_pcs(eigenvectors,variance_list,total_var,sys.argv[2])

print('The dimensions that are required capture α=0.95 fraction of the total variance are {}'.format(len(eigenvalues)-top_values))

first_pcs = eigenvectors[:,-1:]
second_pcs = eigenvectors[:,-2:-1]

first_two_pcs = np.concatenate([first_pcs,second_pcs],axis=1)
print(first_two_pcs)

projection = np.dot(kbar,first_two_pcs)  ### row vector

plt.figure(figsize=(20,12))
plt.scatter(projection[:,0], projection[:,1], alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Projections on two dimension')
plt.show()




mean_vector = np.array(np.mean(D, axis=0))

Dbar = D - mean_vector

Dcov = np.cov(Dbar.T)

eigenvalues2, eigenvectors2 = np.linalg.eigh(Dcov)

first_pcs2 = eigenvectors2[:,-1:]
second_pcs2 = eigenvectors2[:,-2:-1]

first_two_pcs2 = np.concatenate([first_pcs2,second_pcs2],axis=1)

projections2 = np.dot(Dbar,first_two_pcs2)

plt.figure(figsize=(20,12))
plt.scatter(projections2[:,0], projections2[:,1], alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Projections on two dimension')
plt.show()


norm2_list = []
for i in range(len(D)):
    norm2 = np.linalg.norm(D[i])**2
    norm2_list.append(norm2)

S = np.array(norm2_list)

S = np.mat(S)

L = np.dot(D,D.T)

A = S + S.T - 2*L



def projection(matrix,spread):
    spread = float(spread)
    G = np.exp(-matrix/(spread))
    Gbar = np.dot(np.dot(m1,G),m1)
    Geigenvalues, Geigenvectors = np.linalg.eigh(Gbar)
    Gvariance_list, Gtotal_var,Geigenvectors = variance(Geigenvalues,Geigenvectors)
    first_Gpcs = Geigenvectors[:,-1:]
    second_Gpcs = Geigenvectors[:,-2:-1]
    first_two_Gpcs = np.concatenate([first_Gpcs,second_Gpcs],axis=1)
    Gprojections = np.dot(Gbar,first_two_Gpcs)
    plt.figure()
    x2 = np.array(Gprojections[:,0])
    y2 = np.array(Gprojections[:,1])
    plt.scatter(x2, y2)
    plt.title('Projections on two dimension')
    plt.show()
projection(A,sys.argv[3])
