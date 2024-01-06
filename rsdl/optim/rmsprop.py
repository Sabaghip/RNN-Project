from rsdl.optim import Optimizer
import numpy as np

# TODO: implement RMSprop optimizer like SGD
class RMSprop(Optimizer):
    def __init__(self, layers, learning_rate=0.1,epsilon = 1, decay_rate=0.9):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.decay_rate
        self.v_ = 0

    def step(self):
        # TODO: update weight and biases ( Don't use '-=' and use l.weight = l.weight - ... )
        for l in self.layers:
            temp = l.parameters()
            params = temp[0]
            bias = temp[1]
            self.v_ = self.decay_rate * self.v_ + (1 - self.decay_rate) * (params.grad**2)
            l.weight = l.weight - self.learning_rate * params.grad *(1/(np.sqrt(self.v_) + self.epsilon))
            if l.need_bias:
                l.bias = l.bias - self.learning_rate * bias.grad
             
            
            