---
layout: single
title:  "Going Back-to-Basics -- Convolutional Neural Network from Scratch"
date:   2019-03-08 12:07:56 -0700
categories: Deep Learning, CNN

---




<link rel="stylesheet" type="text/css" href="../semantic/semantic.min.css">
<script
src="https://code.jquery.com/jquery-3.1.1.min.js"
integrity="sha256-hVVnYaiADRTO2PzUGmuLJr8BLUSjGIZsDYGmIJLv2b8="
crossorigin="anonymous"></script>
<script src="../semantic/semantic.min.js">
</script>



<br>





## Going Back-to-Basics -- Convolutional Neural Network from Scratch


```python
import numpy as np
import h5py
import time
import copy
from random import randint
import argparse
```


```python
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
#reshaping data
MNIST_data.close() 
```


```python
#Implementation of stochastic gradient descent algorithm
#number of inputs
#--lr=0.01 --epochs=20 --k_num=30 --k_size=7 --ps 3500
LR = 0.01
num_epochs = 20
num_inputs = 28*28
#number of outputs
num_outputs = 10
#hidden layers
model = {}
image_lw = 28
#number of channels
NUM_CHANNELS = 5
#kernel size
kx, ky = 5,5
# Assume square kernel
out_dim_size = image_lw - kx + 1
#training a on a partial set
ps = 3500
```


```python
#Xavier initalization of Weight, Convlutional Stack and Bias
model['W'] = np.random.randn(num_outputs,out_dim_size,out_dim_size,NUM_CHANNELS)/ np.sqrt(image_lw**2)
model['K'] =  np.random.randn(kx,ky,NUM_CHANNELS) / np.sqrt(kx*ky)
model['b1'] = np.random.randn(num_outputs) / np.sqrt(num_outputs)
#model gradients
model_grads = copy.deepcopy(model) 
```


```python
#softmax function
def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

#ReLU function
def relu(z):
    return np.maximum(z,0)

#ReLU derivative
def relu_p(z):
    z[z >= 0] = 1
    z[z < 0] = 0
    return z
```


```python
def im2col(x,outdx,outdy,kern_x, kern_y):
    """
    input: x image dim 28*28 minst()
    output: (kx*ky, 28-kx+1, 28 - kx +1 )
    """
    x_out = np.zeros((kern_x*kern_y,outdx*outdy))
    x_out_col = 0
    for i in range(outdx):
        for j in range(outdy):
            x_out[:,x_out_col] = x[i:i+kern_x,j:j+kern_y].flatten()
            x_out_col += 1
    return x_out

```


```python
def kernal2mat(kern,kern_x, kern_y, kern_d):
    """
    input: kernel kx X ky dim
    output: #number of channels, kx * ky 
    """
    kern_out = np.zeros((kern_d, kern_x*kern_y))
    for i in range(kern_d):
        kern_out[i] = kern[:,:,i].T.flatten()
    return kern_out
```


```python
#Convolution operation
def convolve(x,kern):
    """
    #x is the input image
    output: (w,h,channels)
    """
    kern_x, kern_y, kern_d = kern.shape
    outdx, outdy = x.shape[1] - kern_x +1, x.shape[0] - kern_y +1
    x_mat = im2col(x,outdx, outdy, kern_x, kern_y)
    k_matrix = kernal2mat(kern,kern_x, kern_y,kern_d)
    z = np.dot(k_matrix,x_mat)
    return np.reshape(z.T,(outdx, outdy, kern_d))
```


```python
#forward pass
def forward(x,y, model):
    #im2col method
    Z = convolve(x,model["K"])
    H = relu(Z)
    U = np.zeros(num_outputs)
    for i in range(NUM_CHANNELS):
        U += np.sum(
            np.multiply(model["W"][:,:,:,i],H[:,:,i]), axis=(1,2))\
        +model["b1"]
    F = softmax_function(U)
    return F, H, U, Z
```


```python
#backward pass
def backward(x,y, model, F, H, U, Z):
    indicator = np.zeros(num_outputs)
    indicator[y] = 1
    drdu = - (indicator - F)
    drdb = drdu
    
    drdw = np.zeros((num_outputs,out_dim_size,out_dim_size,NUM_CHANNELS))
    delta = np.zeros((out_dim_size,out_dim_size,NUM_CHANNELS))
    for i in range(num_outputs):
        drdw[i,:,:,:] = drdu[i]*H
        delta += drdu[i]*model["W"][i]
    
    drdk = convolve(x,np.multiply(relu_p(Z),delta))
    model_grads["W"]  = drdw
    model_grads["b1"] = drdb
    model_grads["K"]  = drdk
    return model_grads

```


```python
#Training Loop
print("!"*40)
print("!"*15 + " TRAINING " + "!"*15)
print("!"*40)

import time
time1 = time.time()

x_train_p = x_train[0:ps]
y_train_p = y_train[0:ps]

for epochs in range(num_epochs):
    total_correct = 0
    losses = np.zeros(len(x_train_p))
    for n in range(len(x_train_p)):
        n_random = randint(0,len(x_train_p)-1)
        y = y_train_p[n_random]
        x = x_train_p[n_random][:]
        x = np.reshape(x,(-1,28))
        #forward
        F, H, U, Z = forward(x, y, model)
        #backwards
        model_grads = backward(x,y, model, F, H, U, Z)
        #update grads
        model["K"] -= LR*model_grads["K"]
        model["W"] -= LR*model_grads["W"]
        model["b1"] -= LR*model_grads["b1"]
        
        prediction = np.argmax(F)
        if (prediction == y):
            total_correct += 1
            
            
    print("Epoch: ", epochs, "Training:", total_correct/np.float(len(x_train_p)) )
```

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!! TRAINING !!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Epoch:  0 Training: 0.8322857142857143
    Epoch:  1 Training: 0.908
    Epoch:  2 Training: 0.8925714285714286
    Epoch:  3 Training: 0.8271428571428572
    Epoch:  4 Training: 0.8777142857142857
    Epoch:  5 Training: 0.904
    Epoch:  6 Training: 0.9274285714285714
    Epoch:  7 Training: 0.926
    Epoch:  8 Training: 0.9448571428571428
    Epoch:  9 Training: 0.9417142857142857
    Epoch:  10 Training: 0.95
    Epoch:  11 Training: 0.956
    Epoch:  12 Training: 0.9625714285714285
    Epoch:  13 Training: 0.9634285714285714
    Epoch:  14 Training: 0.9548571428571428
    Epoch:  15 Training: 0.9562857142857143
    Epoch:  16 Training: 0.964
    Epoch:  17 Training: 0.9711428571428572
    Epoch:  18 Training: 0.9737142857142858
    Epoch:  19 Training: 0.9562857142857143
    Epoch:  20 Training: 0.97
    Epoch:  21 Training: 0.9702857142857143
    Epoch:  22 Training: 0.9728571428571429
    Epoch:  23 Training: 0.9814285714285714
    Epoch:  24 Training: 0.9777142857142858
    Epoch:  25 Training: 0.9691428571428572
    Epoch:  26 Training: 0.9525714285714286
    Epoch:  27 Training: 0.9634285714285714
    Epoch:  28 Training: 0.9582857142857143
    Epoch:  29 Training: 0.9842857142857143
    Epoch:  30 Training: 0.9911428571428571
    Epoch:  31 Training: 0.9885714285714285
    Epoch:  32 Training: 0.9954285714285714
    Epoch:  33 Training: 0.9954285714285714
    Epoch:  34 Training: 0.9985714285714286
    Epoch:  35 Training: 0.9988571428571429
    Epoch:  36 Training: 0.9937142857142857
    Epoch:  37 Training: 0.986
    Epoch:  38 Training: 0.9934285714285714
    Epoch:  39 Training: 0.992
    Epoch:  40 Training: 0.9908571428571429
    Epoch:  41 Training: 0.9834285714285714
    Epoch:  42 Training: 0.9488571428571428
    Epoch:  43 Training: 0.9691428571428572
    Epoch:  44 Training: 0.9517142857142857
    Epoch:  45 Training: 0.9714285714285714
    Epoch:  46 Training: 0.9734285714285714
    Epoch:  47 Training: 0.972
    Epoch:  48 Training: 0.9605714285714285
    Epoch:  49 Training: 0.9662857142857143



```python
#Testing the super overfit model 
#84% accuracy 
print("~"*40)
print("~"*16 + " TESTING " + "~"*16)
print("~"*40)



total_correct = 0
for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    x = np.reshape(x,(-1,28))
    p, _, _, _ = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print("Testing:", total_correct/np.float(len(x_test) ) )

```

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~ TESTING ~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Testing: 0.8459

