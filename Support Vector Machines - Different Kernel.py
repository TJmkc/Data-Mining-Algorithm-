import pandas as pd
import numpy as np

import sys

filename = str(sys.argv[1])

loss = str(sys.argv[2])

c_svm = int(sys.argv[3])

eps = float(sys.argv[4])

maxiter = int(sys.argv[5])


kernel_name = str(sys.argv[6])

if kernel_name != 'linear':
    kernel_param = list(sys.argv[7])
    v = len(kernel_param)
    if v > 1:
        q = int(kernel_param[0])
        c = int(kernel_param[-1])
    else:
        spread = int(kernel_param[0])


energy = pd.read_csv(sys.argv[1])



### label the target variable, -1 / 1

data = energy.drop(columns = ['date','rv2'])
data.loc[data['Appliances'] <= 50,'Appliances'] = 1

data.loc[data['Appliances'] > 50,'Appliances'] = -1

### Apply centralization

f = lambda x: (x - np.max(x)) / ((np.max(x)- np.min(x)))

x_data = data.iloc[:,1:].apply(f)

regularize_data = data.iloc[:,:1].join(x_data)

## take 5000 of the data points

matrix = regularize_data.values
matrix = matrix[:5000]

np.random.shuffle(matrix)  ## shuffle the dataset

## train, test split by 70% - 30%

split_line = int(0.7 * len(matrix))
train = matrix[:split_line]
test  = matrix[split_line+1:]

train_x  = train[:,1:]    ## Attribute
train_y  = train[:,0] ## One hot encode the response variable

test_x  = test[:,1:]
test_y  = test[:,0]


## Define kernel function


def hinge_linear(x):
    L_kernel = np.dot(x, x.T)
    return np.array(L_kernel)


def hinge_Gaussian(x, spread):
    norm2_list = []
    for i in range(len(x)):
        norm2 = np.linalg.norm(x[i]) ** 2
        norm2_list.append(norm2)

    S = np.array(norm2_list)
    S = np.mat(S)
    L = np.dot(x, x.T)
    A = S + S.T - 2 * L

    G_kernel = np.exp(-A / (2 * (spread)))
    return np.array(G_kernel)


def hinge_polynomial(x, q, c_kernel):
    P_kernel = (c_kernel + np.dot(x, x.T)) ** q
    return np.array(P_kernel)


def quadratic_linear(x, c_svm):
    loss = (1 / (2 * c_svm)) * (np.eye(len(x), len(x)))
    L_kernel = np.dot(x, x.T) + loss
    return np.array(L_kernel)


def quadratic_Gaussian(x, spread, c_svm):
    loss = (1 / (2 * c_svm)) * (np.eye(len(x), len(x)))
    norm2_list = []
    for i in range(len(x)):
        norm2 = np.linalg.norm(x[i]) ** 2
        norm2_list.append(norm2)

    S = np.array(norm2_list)
    S = np.mat(S)
    L = np.dot(x, x.T)
    A = S + S.T - 2 * L

    G_kernel = np.exp(-A / (2 * (spread))) + loss
    return np.array(G_kernel)


def quadratic_polynomial(x, q, c_kernel, c_svm):
    loss = (1 / (2 * c_svm)) * (np.eye(len(x), len(x)))
    P_kernel = (c_kernel + np.dot(x, x.T)) ** q + loss
    return np.array(P_kernel)

## Return kernel matrix


def kernel_matrix(D,loss,kernel,c_svm):
    if loss == 'hinge':
        if kernel == 'linear':
            kernel_matrix = hinge_linear(D)
        elif kernel == 'gaussian':
            kernel_matrix = hinge_Gaussian(D,spread)
        else:
            kernel_matrix = hinge_polynomial(D,q,c)
    else:
        if kernel == 'linear':
            kernel_matrix = quadratic_linear(D,c_svm)
        elif kernel == 'gaussian':
            kernel_matrix = quadratic_Gaussian(D,spread,c_svm)
        else:
            kernel_matrix = quadratic_polynomial(D,q,c,c_svm)
    return kernel_matrix

## Augmented kernel matrix by adding 1 in each attribute

def augment_K(kernel):
    k0 = kernel
    aug_K = k0 + 1
    return aug_K

### SVM Stochastic gradient ascent


def svm(D, loss, c_svm, eps, maxiter,kernel):
  #print(k)
  ## Augmented the kernel matrix
    kernel_m = kernel_matrix(D,loss,kernel,c_svm)
    k_head = augment_K(kernel_m)
    n = len(k_head)

# Set step size
    step_size = []
    for p in range(n):
        step_size.append(1/k_head[p,p])

# Initial alpha  and iteration counter t
    alpha = []

    alpha0 = [0] * n

    alpha.append(alpha0)

    t = 0

## svm repeat loop & stochastic gradient descent
    while True:
        #print(t)
        middle_a = alpha[t].copy()
        randomlist = np.random.choice(n,n,replace = False)  ## shuffle the index
        for index in randomlist:

            sum1 = np.dot((middle_a * train_y).T,k_head[:,index])
            gradient  = 1 - (train_y[index] * sum1)
            b = step_size[index] * gradient
            middle_a[index] = middle_a[index] + b
            if middle_a[index] < 0:
                middle_a[index] = 0
            if loss == 'hinge':
                if middle_a[index] > c_svm:
                    middle_a[index] = c_svm

        alpha.append(middle_a)
        t = t+1


        a_previous = np.array(alpha[t-1])
        a_current = np.array(alpha[t])
        diff = np.linalg.norm(a_current - a_previous)

        if t > maxiter:
            return alpha[t],k_head
            break
        if diff <= eps:
            return alpha[t],k_head
            break


## Calculate the number of support vector

def num_sv(y_pred, alpha, loss, c_svm):
    pred_y = y_pred
    a1 = alpha
    sv = 0 # Number of support vector
    if loss == 'hinge':
        for i in range(len(a1)):
            if a1[i] > 0 and a1[i] < c_svm :
                sv = sv +1
    else:
        for i in range(len(a1)):
            if a1[i] > 0:
                sv = sv +1
    corr = 0
    for i in range(len(test_y)):
        if pred_y[i] == test_y[i]:
            corr += 1
    acc = corr / len(test_y)
    print('Accuracy is ' + str(acc))
    print('The number of support vector is ' + str(sv))
    return acc, sv

### Different kernel prediction


def linear_prediction(alpha,loss,c_svm):
    w = 0
    b = 0
    k = 0
    a1 = alpha
    if loss == 'hinge':
        for i in range(len(train_x)):  ## Calculating Weight vector W
            if a1[i] > 0:
                w += a1[i] * train_y[i] * train_x[i]
        for i in range(len(train_x)):
            if a1[i] < c_svm:          ## Calculating bias b, ignoring alpha = c
                b += train_y[i] - np.dot(w.T,train_x[i])
    else:
        for i in range(len(train_x)):
            if a1[i] > 0:
                w += a1[i] * train_y[i] * train_x[i]
        for i in range(len(train_x)):
            if a1[i] > 0:
                b += train_y[i] - np.dot(w.T,train_x[i])
    pred_y = []
    for points in test_x:
        sum2 = 0
        for i in range(len(a1)):
            if a1[i] > 0:
                sum2 += a1[i] * train_y[i] * (np.dot(train_x[i].T,points) + 1)
        pred = np.sign(sum2)
        pred_y.append(pred)
    acc,sv = num_sv(pred_y, a1, loss, c_svm)
    b = b / sv
    print('The weight vector is ' + str(w))
    print('The bias is ' + str(b))
    return acc, sv, w, b

def gaussian_prediction(alpha,spread,loss,c_svm):
    a1 = alpha
    pred_y = []
    for points in test_x:
        sum2 = 0
        for i in range(len(a1)):
            d = -((np.linalg.norm(train_x[i] - points)))
            k_g = np.exp(d/(2*(spread))) + 1
            sum2 += a1[i] * train_y[i] * (k_g)
        pred = np.sign(sum2)
        pred_y.append(pred)
    acc,sv = num_sv(pred_y, a1, loss, c_svm)
    return acc, sv

def polynomial_prediction(alpha, q, c, loss, c_svm):
    a1 = alpha
    pred_y = []
    for points in test_x:
        sum2 = 0
        for i in range(len(a1)):
            d = (c + np.dot(train_x[i].T, points))**q
            k_g = d + 1
            sum2 += a1[i] * train_y[i] * (k_g)
        pred = np.sign(sum2)
        pred_y.append(pred)
    acc,sv = num_sv(pred_y, a1, loss, c_svm)
    return acc,sv


a1,k1 = svm(train_x,loss,c_svm,eps,maxiter,kernel_name)

if kernel_name == 'linear':
    linear_prediction(a1, loss, c_svm)
elif kernel_name == 'gaussian':
    gaussian_prediction(a1, spread, loss, c_svm)
else:
    polynomial_prediction(a1, q, c, loss, c_svm)




