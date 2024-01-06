from rsdl import Tensor
import numpy as np

def MeanSquaredError(preds: Tensor, actual: Tensor):
    # TODO : implement mean squared error
    err = ((actual - preds) * (actual - preds)).sum()
    res = err * (Tensor(1 / len(actual.data), requires_grad=True, depends_on=actual.depends_on))
    return res

def CategoricalCrossEntropy(preds: Tensor, actual: Tensor):
    # TODO : imlement categorical cross entropy
    sum = 0
    for i in range(len(preds.data)):
        sum -= actual.data[i] * np.log(preds.data[i]) 
    return None



