import numpy as np
import pandas as pd
import random
import sys


data = pd.read_csv(sys.argv[1])

regression_data = data.drop(columns = ['date','rv2'])


regression_data.loc[regression_data['Appliances'] <= 30,'Appliances'] = 1

regression_data.loc[(regression_data['Appliances'] > 30) & (regression_data['Appliances'] <= 50),'Appliances'] = 2

regression_data.loc[(regression_data['Appliances'] > 50) & (regression_data['Appliances'] <= 100),'Appliances'] = 3

regression_data.loc[regression_data['Appliances'] > 100,'Appliances'] = 4

sample = int(0.7*len(regression_data))
training = regression_data[:sample]
test = regression_data[sample:]

response_ytrain = np.array(pd.get_dummies(data = training['Appliances']))

response_ytest = np.array(pd.get_dummies(data = test['Appliances']))

train_x = training.iloc[:,1:]

x0 = np.array([1]*len(train_x))
x0 = np.mat(x0)

D = train_x.values

aug_trainX = np.insert(D,0,values=x0,axis=1)

test_x = test.iloc[:,1:]
xtest = test_x.values
x1 = np.array([1]*len(test_x))
x1 = np.mat(x1)
aug_xtest = np.insert(xtest,0,values=x1,axis=1)


def SGA(D, eta, eps, maxiter):
    m, n = np.shape(D)  ### m points, n dimension of the augmented dataset
    Initial_W = np.zeros((n, 4))  ### Initial vector of W
    t = 0  # iteration counter t
    random_number = np.random.choice(m, m, replace=False)  ## random number list --- make the point random
    all_W = []  ### W matrix, record W in each iteration
    all_W.append(Initial_W)  ## Append Initial W vector in t = 0 iteration
    while True:
        w_head = all_W[t].copy()  ##Before iteration,  make a copy of the current W
        for i in range(m):  ### For each point in the dataset
            x = D[random_number[i], :]  ### Choose the point X randomly
            for k in range(0, 3):  ### for each class k, do gradient ascend
                sigma = (np.exp(np.dot(w_head[:, 0].T, x) - np.dot(w_head[:, k].T, x)) +
                         np.exp(np.dot(w_head[:, 1].T, x) - np.dot(w_head[:, k].T, x)) +
                         np.exp(np.dot(w_head[:, 2].T, x) - np.dot(w_head[:, k].T, x)) +
                         np.exp(np.dot(w_head[:, 3].T, x) - np.dot(w_head[:, k].T, x)))
                ceta = 1 / sigma
                gradient = (response_ytrain[random_number[i]][k] - ceta) * x  ### calculate the gradient,
                w_head[:, k] = w_head[:, k] + eta * gradient  ## update the copy matrix W_head in each point

        all_W.append(w_head)  ## After all points are seen, store the weight w_head in all_W[t+1]
        t = t + 1  ## Update the iteration counter
        sum1 = 0  ## calculate the sum of the distance between W[t] and W[t-1]
        for j in range(0, 4):
            sum1 += np.linalg.norm(all_W[t][:, j] - all_W[t - 1][:, j])
        if sum1 <= eps:  ## if W less and equal than the threshold, converge, return the Weight Matrix W[t]
            print(sum1)
            print(w_head)
            return all_W[t], t
            break
        if t > maxiter:  ### if the iteration time greater than the max iteration time, not converge,return the W[t]
            print(sum1)
            print(w_head)
            return all_W[t], t
            break

w,t = SGA(aug_trainX,sys.argv[2],sys.argv[3],sys.argv[4])

pred_y  = []
for x in aug_xtest:
    c = []
    for k in range(0,4):
        pi = np.exp(np.dot(w[:,k].T,x))
        sigma = (np.exp(np.dot(w[:,0].T,x)) +
                     np.exp(np.dot(w[:,1].T,x)) +
                     np.exp(np.dot(w[:,2].T,x)))
        if k != 3:
            ceta = pi / (1+sigma)
            c.append(ceta)
        else:
            ceta = 1 / (1+sigma)
            c.append(ceta)
    for i,prob in enumerate(c):
        if prob == max(c):
            pred_y.append(i+1)

pred_y = list(pred_y)

yi = list(test['Appliances'])

corr = 0
for i in range(len(yi)):
    if pred_y[i] == yi[i]:
        corr += 1

acc = corr/ len(yi)
acc,corr
