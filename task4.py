# Task 4
import numpy as np

from rsdl import Tensor
from rsdl.losses import loss_functions

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 6

# TODO: define w and b (y = w x + b) with random initialization ( you can use np.random.randn )
w = np.array([np.random.randn(), np.random.randn(), np.random.randn()]).reshape(3,1)
b = np.random.randn()

print(w)
print(b)

learning_rate = 0.002
batch_size = 5

for epoch in range(100):
    
    epoch_loss = 0.0
    
    for start in range(0, 100, batch_size):
        end = start + batch_size

        inputs = X[start:end]
        

        # TODO: predicted
        predicted = inputs @ w + b
        

        actual = y[start:end]
        actual.data = actual.data.reshape((batch_size,1))
        # TODO: calcualte MSE loss
        
        # TODO: backward
        # hint you need to just do loss.backward()

        epoch_loss += loss_functions.MeanSquaredError(predicted, actual)


        # TODO: update w and b (Don't use 'w -= ' and use ' w = w - ...') (you don't need to use optim.SGD in this task)
        # print(predicted - actual)
        # print(w)
        inputs.data = inputs.data.T
        w = w - learning_rate * 2 * (inputs @ (predicted - actual)).data/batch_size
        b = b - learning_rate * 2 * (predicted - actual).sum().data/batch_size

print(w)
print(b)

