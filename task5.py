# Task 5
import numpy as np

from rsdl import Tensor
from rsdl.layers import Linear
from rsdl.optim import SGD
from rsdl.losses import loss_functions

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 5

# TODO: define a linear layer using Linear() class  
l = Linear(3, 1, True)

# TODO: define an optimizer using SGD() class 
learning_rate = 0.007
optimizer = SGD([l], learning_rate)

# TODO: print weight and bias of linear layer
print(l.weight)
print(l.bias)



batch_size = 20

for epoch in range(100):
    
    epoch_loss = 0.0
    
    for start in range(0, 100, batch_size):
        l.zero_grad()
        end = start + batch_size

        inputs = X[start:end]

        # TODO: predicted
        predicted = l(inputs)

        actual = y[start:end]
        actual.data = actual.data.reshape(batch_size, 1)
        # TODO: calcualte MSE loss
        loss =loss_functions.MeanSquaredError(predicted, actual)
        
        
        # TODO: backward
        # hint you need to just do loss.backward()
        loss.backward()
        # print("new params", params)
        # TODO: add loss to epoch_loss
        epoch_loss += loss


        # TODO: update w and b using optimizer.step()
        optimizer.step()
        

# TODO: print weight and bias of linear layer
 
print(l.weight)
print(l.bias)