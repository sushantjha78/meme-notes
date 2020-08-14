
import os
from PIL import Image
from random import shuffle, choice
import numpy as np
import os
from math import floor
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean


def load_images_resize_bw(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            img = np.array(img.convert('L'))
            img = resize(img, (200, 100), anti_aliasing=True)
            img = img.flatten()
            img = np.reshape(img.flatten(), (len(img), 1))
            images.append(img)
    return images

def label_images(image_list, value):
    labelled_image = []
    for i in range(len(image_list)):
        labelled_image.append(np.append(image_list[i], value))
    return labelled_image




x = label_images(notes, 1)
x += label_images(memes, 0)
shuffle(x)
train_range = round(0.92*len(x)) # 92% for train
x_train = x[:train_range][:]
x_test = x[train_range:][:]


nn_architecture = [
    {"input_dim": 20000, "output_dim": 1000, "activation": "relu"},
    {"input_dim": 1000, "output_dim": 100, "activation": "relu"},
    {"input_dim": 100, "output_dim": 10, "activation": "relu"},
    {"input_dim": 10, "output_dim": 2, "activation": "relu"},
    {"input_dim": 2, "output_dim": 1, "activation": "sigmoid"},
]


def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        params_values['W' + str(layer_idx)] = np.random.randn( layer_output_size, layer_input_size) * (np.sqrt(2/layer_input_size))
        params_values['b' + str(layer_idx)] = np.full((layer_output_size, 1), 0.1)
        
    return params_values

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    Z = np.array(Z)
    Z[Z <= 0] = 0
    dZ = np.array(Z, copy = True)

    return dZ;

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
        
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    
    memory = {}
    A_curr = X
    
    for idx, layer in enumerate(nn_architecture):
        A_curr = np.array(A_curr)
        mean = A_curr.mean(axis = 0)
        var = np.var(A_curr, axis = 0)
        A_curr = (A_curr - mean)/(np.sqrt(var + 0.000001))
        
        layer_idx = idx + 1
        A_prev = A_curr
        
        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    return A_curr, memory

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = 1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

def convert_prob_into_class(probs):
         probs_ = np.copy(probs)
         probs_[probs_ > 0.5] = 1
         probs_[probs_ <= 0.5] = 0
         return probs_

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]
    
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward 
    
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
    layer_idx_curr = len(nn_architecture)
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
    for layer in reversed(list(nn_architecture)):
        layer_idx_prev = layer_idx_curr - 1
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
        layer_idx_curr -= 1
    
    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        layer_idx += 1
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values;

def train(x_train, minibatch_size, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []
    
    for i in range(epochs):
        shuffle(x_train)
        for n in range(floor(len(x_train) / minibatch_size)):
            start = n*minibatch_size
            X = []
            Y = []
            for i in range(minibatch_size):
                X.append(x_train[start + i][:20000])
                Y.append(x_train[start + i][20000])
            X = np.array(X).T
            Y = np.array(Y)
            Y = (np.reshape(Y, (minibatch_size, 1))).T

            Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
            cost = get_cost_value(Y_hat, Y)
            cost_history.append(cost)
            accuracy = get_accuracy_value(Y_hat, Y)
            accuracy_history.append(accuracy)
            
            grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
            params_values = update(params_values, grads_values, nn_architecture, learning_rate)
            

        if (len(x_train) % minibatch_size) != 0:
            start = (n+1)*minibatch_size
            X = []
            Y = []
            for i in range(len(x_train) % minibatch_size):
                X.append(x_train[start + i][:20000])
                Y.append(x_train[start + i][20000])
            X = np.array(X).T
            Y = np.array(Y)
            Y = (np.reshape(Y, (len(x_train) % minibatch_size, 1))).T
            print("----------------------------------------------------------------------------------")

            Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
            cost = get_cost_value(Y_hat, Y)
            cost_history.append(cost)
            accuracy = get_accuracy_value(Y_hat, Y)
            accuracy_history.append(accuracy)
            
            grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
            params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
    return cost_history, accuracy_history




 
train(x_train, 64, nn_architecture, 10, 0.05)
