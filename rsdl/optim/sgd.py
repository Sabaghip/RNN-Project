from rsdl.optim import Optimizer

# TODO: implement step function
class SGD(Optimizer):
    def __init__(self, layers, learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        # TODO: update weight and biases ( Don't use '-=' and use l.weight = l.weight - ... )
        for l in self.layers:
            temp = l.parameters()
            params = temp[0]
            bias = temp[1]
            l.weight = l.weight - self.learning_rate * params.grad
            if l.need_bias:
                l.bias = l.bias - self.learning_rate * bias.grad
