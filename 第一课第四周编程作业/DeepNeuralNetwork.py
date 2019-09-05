"""
构建一个两层神经网络和一个多层神经网络
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import lr_utils
from testCases import *
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward

np.random.seed(1)

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

#===================================Initialization==========================================
# 2-layer Neural Network
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b'+str(l)] = np.zeros(shape = (layer_dims[l],1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

#=====================================forward propagation===================================
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W,A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z,linear_cache = linear_forward(A_prev, W, b)
        A,activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,parameters['W'+str(l)],
                                             parameters['b'+str(l)],activation='relu')
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A,parameters['W'+str(L)],
                                         parameters['b'+str(L)],activation='sigmoid')
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches

#======================================cost function=======================================
def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = -(1/m)*np.sum(np.multiply(Y,np.log(AL))+np.multiply((1-Y),np.log(1-AL)))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost

#====================================backward propagation==================================
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ,cache[0].T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev = np.dot(cache[1].T,dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)

    # Shorten the code
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients.
    # Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                            activation="sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches".
        # Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        grads["dA"+str(l+1)], grads["dW"+str(l+1)], grads["db"+str(l+1)] = linear_activation_backward(grads["dA"+str(
            l+2)],current_cache,activation="relu")

    return grads

#=======================================Update Parameters==================================
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

#===================================put together==========================================
def two_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=2500, print_cost=False):
    """
    two layers model

    Arguments:
        X: data set(n,m)
        Y: labels(1,m)
        layer_dims:(n_x,n_h,n_y)
                   n_x -- size of the input layer
                   n_h -- size of the hidden layer
                   n_y -- size of the output layer
        learning_rate: 学习速率
        num_iterations: Number of iterations in gradient descent loop
        print_cost: if True, print the cost every 1000 iterations

    Returns:
        parameters: parameters learnt by the model. They can then be used to predict.

    """

    np.random.seed(2)
    m = X.shape[1]   # number of dataset
    (n_x, n_h, n_y) = layer_dims
    grads = {}
    costs = []

    #initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Loop (gradient descent)
    for i in range(0,num_iterations):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        #forward propagation
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")

        #cost function
        cost = compute_cost(A2,Y)

        #backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")
        grads["dW2"] = dW2
        grads["dW1"] = dW1
        grads["db2"] = db2
        grads["db1"] = db1

        #Gradient descent parameter updata
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 1000 iterations
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))
    #plot cost
    plt.plot(np.squeeze(costs))
    plt.xlabel("iterations(per tens)")
    plt.ylabel("cost")
    plt.title("learing rate="+str(learning_rate))
    plt.show()

    return parameters

def L_layer_model(X, Y, layer_dims, learning_rate=0.0075,num_iterations=2500,print_cost=False):
    """
    L layers

    Args:
        X: data set of shape(n,m)
        Y: labels of shape(1,m)
        layer_dims: python array (list) containing the dimensions of each layer in our network
        num_iterations: number of iterations in gradient descent loop
        print_cost: if ture,print the cost every 1000 iterations

    Returns:
        parameters: parameters learnt by the model.They can then be used to predict.

    """
    np.random.seed(3)
    m = X.shape[1]
    costs = []

    #initialize parameters
    parameters = initialize_parameters_deep(layer_dims)

    #Loop
    for i in range(num_iterations):
        #forward propagation
        AL, caches = L_model_forward(X, parameters)

        #cost function
        cost = compute_cost(AL, Y)

        #backward propagation
        grads = L_model_backward(AL, Y, caches)

        #Gradient decent parameters update
        parameters = update_parameters(parameters, grads, learning_rate)

        #Print the cost in every 1000 iterations
        if i%100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration %i:%f"% (i,cost))

    #plot cost
    plt.plot(np.squeeze(costs))
    plt.xlabel("iterations(per tens)")
    plt.ylabel("cost")
    plt.title("learing rate="+str(learning_rate))
    plt.show()

    return parameters

#======================================predict============================================
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    AL, caches = L_model_forward(X,parameters)
    predictions = np.round(AL)
    # np.round()返回参数的四舍五入值

    return predictions


# Build a model with a n_h-dimensional hidden layer
layer_dims = (train_x.shape[0],20,7,5,train_y.shape[0])
parameters = L_layer_model(train_x, train_y,layer_dims, print_cost=True)

# Print accuracy
predictions = predict(parameters, test_x)
print ('Accuracy: %d' % float((np.dot(test_y, predictions.T)
            + np.dot(1 - test_y, 1 - predictions.T)) / float(test_y.size) * 100) + '%')
