import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
data = pd.read_csv(sys.argv[1])

data = data.drop(columns = ['date','rv2'])

sample = int(round(0.7*len(data),0))

subdata = data[:sample]

y = subdata['Appliances']

y = np.array(y)

x = subdata.iloc[:,1:]

D = x.values

x0 = np.array([1]*len(x))
x0 = np.mat(x0)

def gram_schmidt(A):
    Q = np.zeros_like(A)

    R = np.zeros([len(A.T),len(A.T)])

    cnt = 0
    for i in range(0,len(A.T)):
        u = np.copy(A.T[i])
        for j in range(0,cnt):
            projection = (np.dot(A.T[i].T, Q[:,j]))/(np.dot(Q[:,j].T,Q[:,j]))
            R[j,i] = projection
            u = u -  np.dot(projection, Q[:,j])

        R[cnt,i]=1
        Q[:,cnt] = u
        cnt += 1
    return Q,R


aug_D = np.insert(D,0,values=x0,axis=1)


Q,R = gram_schmidt(aug_D)


delta = np.zeros([27,27])

for i in range(len(delta.T)):
    square_norm = np.linalg.norm(Q.T[i])**2
    delta[i,i] = square_norm

delta = np.matrix(delta)

In_delta = delta.I

w = np.zeros([27,1])

right_equation = np.array(np.dot(In_delta,np.dot(Q.T,y.T)))


w[-1] = right_equation[0][-1]

for i in range(1,len(w)+1):
    a = 0
    for j in range(1,len(w)+1):
        if i == j:
            continue
        else:
            a += R[-i][-j] * w[-j]
    w[-i] = (right_equation[0][-i] - a)

print(w.T)
print(np.linalg.norm(w))



train_pred_y = np.dot(Q,right_equation.T)

train_y = subdata['Appliances']

train_SSE = 0
for i in range(len(train_y)):
    train_SSE += (train_pred_y[i]-train_y[i])**2
train_MSE = train_SSE  / len(train_y)
print("The MSE of the training set is {}, the SSE of the training set is {}".format(train_MSE,train_SSE))
train_y_mean = sum(train_y) / len(train_y)
train_TSS = 0
for i in range(len(train_y)):
    train_TSS += (train_y[i]-train_y_mean)**2
    train_R2 = (train_TSS-train_SSE) / train_TSS
print("The R2 of the training set is {}".format(train_R2))

test_data = data[sample+1:]

test_y = list(test_data['Appliances'])
test_x = test_data.iloc[:,1:]

x1 = np.array([1]*len(test_x))
x1 = np.mat(x1)

test_x_D = test_x.values
aug_test_x = np.insert(test_x_D,0,values=x1,axis=1)
test_pred_y = np.dot(aug_test_x,w)

test_SSE = 0
for i in range(len(test_y)):
    test_SSE += (test_pred_y[i]-test_y[i])**2
test_MSE = test_SSE  / len(test_y)
print("The MSE of the training set is {}, the SSE of the training set is {}".format(test_MSE,test_SSE))
test_y_mean = sum(test_y) / len(test_y)
test_TSS = 0
for i in range(len(test_y)):
    test_TSS += (test_y[i]-test_y_mean)**2
test_R2 = (test_TSS-test_SSE) / test_TSS
print("The R2 of the test set is {}".format(test_R2))