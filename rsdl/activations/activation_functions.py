from rsdl import Tensor, Dependency
import numpy as np

def Sigmoid(t: Tensor) -> Tensor:
    # TODO: implement sigmoid function
    # hint: you can do it using function you've implemented (not directly define grad func)
    return Tensor(data=1 / (1 + (-t).exp()), requires_grad=t.requires_grad, depends_on=t.depends_on)

def Tanh(t: Tensor) -> Tensor:
    # TODO: implement tanh function
    # hint: you can do it using function you've implemented (not directly define grad func)
    return Tensor(data=(t.exp() - (-t).exp())/(t.exp() + (-t).exp()), requires_grad=t.requires_grad, depends_on=t.depends_on)

def Softmax(t: Tensor) -> Tensor:
    # TODO: implement softmax function
    # hint: you can do it using function you've implemented (not directly define grad func)
    # hint: you can't use sum because it has not axis argument so there are 2 ways:
    #        1. implement sum by axis
    #        2. using matrix mul to do it :) (recommended)
    # hint: a/b = a*(b^-1)
    sum = t.exp() @ np.ones((t.data.shape[1], 1))
    # print("vvvvvvvvvvvvvvv",sum.data, "qqqqqqqqqqqqqqqqq")
    return t.exp() * Tensor(1/sum.data)

def Relu(t: Tensor) -> Tensor:
    # TODO: implement relu function

    # use np.maximum
    data = np.maximum(np.zeros_like(t.data), t.data)

    req_grad = t.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            # use np.where
            return np.where(t.data > 0, grad,np.zeros_like(grad))
        
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


def LeakyRelu(t: Tensor,leak=0.05) -> Tensor:
    """
    fill 'data' and 'req_grad' and implement LeakyRelu grad_fn 
    hint: use np.where like Relu method but for LeakyRelu
    """
    # TODO: implement leaky_relu function
    
    data = np.where(t.data > 0, t.data, leak * t.data)
    
    req_grad = t.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return np.where(t.data > 0, grad, leak * grad)
        
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
