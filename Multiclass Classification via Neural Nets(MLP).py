import pandas as pd
import numpy as np

import sys


energy = pd.read_csv(sys.argv[1])

data = energy.drop(columns = ['date','rv2'])

data.loc[data['Appliances'] <= 30,'Appliances'] = 1

data.loc[(data['Appliances'] > 30) & (data['Appliances'] <= 50),'Appliances'] = 2

data.loc[(data['Appliances'] > 50) & (data['Appliances'] <= 100),'Appliances'] = 3

data.loc[data['Appliances'] > 100,'Appliances'] = 4


f = lambda x: (x - np.max(x)) / ((np.max(x)- np.min(x)))
x_data = data.iloc[:,1:].apply(f)

regularize_data = data.iloc[:,:1].join(x_data)

matrix = regularize_data.values

matrix = data.values


np.random.shuffle(matrix)  ## shuffle the dataset

split_line = int(0.7*len(matrix))   ### 70% of the data as training set, 30% as test set
train = matrix[:split_line]
test = matrix[split_line+1:]


train_x  = train[:,1:]    ## Attribute
train_y  = np.array(pd.get_dummies(data = train[:,0])) ## One hot encode the response variable


test_x  = test[:,1:]
test_y  = np.array(pd.get_dummies(data = test[:,0]))


## activation Function Relu for hidden layers
def relu(x):
    dx = x
    for t,value in enumerate(dx):
        if value <= 0:
            dx[t] = 0
        else:
            dx[t] = value
    return dx


## activation function Softmax for output layer
def softmax(x):
    da = x
    row_max = da.max()
    da = da - row_max
    soft = []
    sum1 = 0
    for value in da:
        e1 = np.exp(value)
        sum1 += e1
    for v in range(len(da)):
        e2 = np.exp(da[v]) / sum1
        soft.append(e2)
    soft = np.array(soft)
    return soft

## Backpropagation for Relu function, the derivative of Relu towards its argument, net gradient

def relu_backward(x):
    dZ = x
    for u, value in enumerate(dZ):
        if value <= 0:
            dZ[u] = 0
        else:
            dZ[u] = 1
    return dZ


def Feed_Forward(x, w_curr, b_curr):
    ### generate input of current hidden layer， linear combination of weight matrix and bias
    z_linear = np.dot(w_curr.T, x) + b_curr

    ## Output of the hidden layer after activated by activative function
    z_active = relu(z_linear)

    return z_linear, z_active


m, n = np.shape(train_x)
q, w = np.shape(train_y)

# Size of input layer

input_layer = n

# Size of output layer

output_layer = w

# Number of hidden layers

h = int(sys.argv[5])

# Size of hidden layers
Hidden = [int(sys.argv[4])]*h

# Initial the weight matrices and bias vector

bias = []
weight = []

# Bias, weight for hidden layer 1
bias.append(np.random.rand(Hidden[0],1))
weight.append(np.random.rand(n, Hidden[0]))

# Bias, weight for the rest hidden layer 

for i in range(1,h):
    bias.append(np.random.rand(Hidden[i],1))
    weight.append(np.random.rand(Hidden[i],Hidden[i]))

# Weight and bias vector for the output layer

bias.append(np.random.rand(output_layer, 1))
weight.append(np.random.rand(Hidden[-1],output_layer))

## Stochastic Gradient Descent

# Iteration Counter
t = 0

# Max iteration times
maxiter = int(sys.argv[3])

# Store the information of each hidder layers
memory = {}
random_number = np.random.choice(m, m, replace=False)

while (t < maxiter):
    for i in range(m):
        x = train_x[random_number[i]]
        x = x.reshape(len(x), 1)
        # Feed - Forward Phase

        # Input layer
        linear_output = [0] * (h + 1)
        active_output = [0] * (h + 1)
        # Input layer
        z_active = x

        active_output[0] = z_active
        linear_output[0] = z_active

        # Hidden layer
        for j in range(h):
            input_z = z_active
            net_z, z_active = Feed_Forward(input_z, weight[j], bias[j])

            linear_output[j + 1] = net_z
            active_output[j + 1] = z_active
            

        # output layer
        input_o = np.dot(weight[-1].T, z_active) + bias[-1]
        net_o = softmax(input_o)

        linear_output.append(input_o)
        active_output.append(net_o)
       

        # Backpropagate Phase, using cross entroy error and softmax function for multi classification
        netgradient = [0] * (h + 1)

        # output layer
        c = []
        # The net gradient of the output layer is o - y from formulation (25.52)
        netgradient_o = net_o - train_y[random_number[i]].reshape(w, 1)
        netgradient[-1] = netgradient_o
        # Hidden layer
        for k in range(h - 1, -1, -1):
            d_relu = relu_backward(linear_output[k+1])
            c.append(d_relu)
            netgradient[k] = (d_relu *
                              np.dot(weight[k + 1], netgradient[k + 1]))

        # Gradient Descent Step
        step_size = float(sys.argv[2])
        weight_gradient = []
        bias_gradient = []
        for l in range(0, h + 1):
            weight_gradient.append(np.dot(active_output[l], netgradient[l].T))
            bias_gradient.append(netgradient[l])

        for p in range(0, h + 1):
            weight[p] = weight[p] - step_size * weight_gradient[p]
            bias[p] = bias[p] - step_size * bias_gradient[p]
    t = t + 1

# test
pred_y = []

for i in range(len(test_x)):
    x = test_x[i]
    x = x.reshape(len(x),1)
# Feed - Forward Phase
    z_active = x
    for j in range(h):
        input_z = z_active
        net_z, z_active = Feed_Forward(input_z, weight[j], bias[j])
    input_o = np.dot(weight[-1].T,z_active) + bias[-1]
    net_o = softmax(input_o)
    for f, value in enumerate(net_o):
        if value == net_o.max():
            pred_y.append(f+1)

true_y = test[:,0]


corr = 0
for t in range(len(true_y)):
    if pred_y[t] == true_y[t]:
        corr += 1
    else:
        continue
accuracy = corr/len(true_y)

print(weight)
print(bias)
print(accuracy)