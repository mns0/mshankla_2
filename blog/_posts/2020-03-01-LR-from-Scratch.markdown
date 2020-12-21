---
layout: single
title:  "Going Back-to-Basics -- Logistic Regression Solved using SGD from Scratch"
date:   2020-03-05 12:07:56 -0700
categories: Logistic Regression, Stochastic Gradient Descent
---


<link rel="stylesheet" type="text/css" href="../semantic/semantic.min.css">
<script
src="https://code.jquery.com/jquery-3.1.1.min.js"
integrity="sha256-hVVnYaiADRTO2PzUGmuLJr8BLUSjGIZsDYGmIJLv2b8="
crossorigin="anonymous"></script>
<script src="../semantic/semantic.min.js">
</script>


<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<br>








## Going Back-to-Basics -- Logistic Regression solved using SGD from Scratch


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
#Implementation of stochastic gradient descent algorithm
#number of inputs
num_inputs = 28*28
#number of outputs
num_outputs = 10
model = {}
#Xavier initalization of Weight, Convlutional Stack and Bias
model['W1'] = np.random.randn(num_outputs,num_inputs) / np.sqrt(num_inputs)
#model gradients
model_grads = copy.deepcopy(model)

```


```python
#softmax function
def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

#Forward Pass
def forward(x,y, model):
    Z = np.dot(model['W1'], x)
    p = softmax_function(Z)
    return p

#Backwards pass
#note derivates of loss function taken ahead of time
def backward(x,y,p, model, model_grads):
    dZ = -1.0*p
    dZ[y] = dZ[y] + 1.0
    for i in range(num_outputs):
        model_grads['W1'][i,:] = dZ[i]*x
    return model_grads
```


```python
#Training Loop
print("!"*40)
print("!"*15 + " TRAINING " + "!"*15)
print("!"*40)

import time
time1 = time.time()
LR = .01
num_epochs = 20
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
    total_correct = 0
    for n in range( len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        p = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x,y,p, model, model_grads)
        model['W1'] = model['W1'] + LR*model_grads['W1']
    print(total_correct/np.float(len(x_train) ) )
time2 = time.time()
print(time2-time1, "seconds")
```

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!! TRAINING !!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    0.8959333333333334
    0.9126833333333333
    0.9195166666666666
    0.9216
    0.9215
    0.9220666666666667
    0.9278333333333333
    0.9292166666666667
    0.9293
    0.9288
    0.93085
    0.9315666666666667
    0.9313333333333333
    0.93
    0.9316833333333333
    0.9301333333333334
    0.93095
    0.9329
    0.9314333333333333
    0.9308166666666666
    84.81015992164612



```python
#Testing the super overfit model 
#92% accuracy 
######################################################
#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    p = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
        
print("{} accuracy on test ".format(total_correct/np.float(len(x_test) )))
```

    0.9242 accuracy on test 



