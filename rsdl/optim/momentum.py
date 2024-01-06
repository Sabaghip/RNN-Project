from rsdl.optim import Optimizer

# TODO: implement Momentum optimizer like SGD
class Momentum(Optimizer):
    def __init__(self, layers, learning_rate=0.1, momentum = 0.01):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.change = 0.0
        self.momentum = momentum

    def step(self):
        # TODO: update weight and biases ( Don't use '-=' and use l.weight = l.weight - ... )
        for l in self.layers:
            temp = l.parameters()
            params = temp[0]
            bias = temp[1]
            self.change = self.learning_rate * params.grad + self.momentum * self.change
            l.weight = l.weight - self.change
            if l.need_bias:
                l.bias = l.bias - self.learning_rate * bias.grad