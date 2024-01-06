from rsdl import Tensor
from rsdl.layers import initializer

class Linear:

    def __init__(self, in_channels, out_channels, need_bias=True, mode='xavier') -> None:
        # set input and output shape of layer
        self.shape = (in_channels, out_channels)
        self.need_bias = need_bias
        # TODO initialize weight by initializer function (mode)
        self.weight = Tensor(
            data=initializer((in_channels, out_channels), mode=mode),
            requires_grad=True
        )
        # TODO initialize weight by initializer function (zero mode)
        if self.need_bias:
            self.bias = Tensor(
                data=0,
                requires_grad=True
            )

    def forward(self, inp):
        # TODO:implement forward propagation
        if self.need_bias:
            return inp @ self.weight + self.bias
        return inp @ self.weight
    
    def parameters(self):
        if self.need_bias:
            return [self. weight, self.bias]
        return [self. weight]
    
    def zero_grad(self):
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()
            
    def __call__(self, inp):
        return self.forward(inp)
