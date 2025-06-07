import numpy as np
from pico import Tensor

def max(a: Tensor, b: Tensor) -> Tensor:
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return Tensor(data=np.maximum(a.data, b.data), _children=(a, b), _op='functional_max')
    elif isinstance(a, Tensor) and isinstance(b, int):
        b_tensor = Tensor(data=np.full(a.shape, b), _children=(Tensor(b), ), _op='cloning_b')
        return Tensor(data=np.maximum(a.data, b_tensor.data), _children=(a, b), _op='functional_max')

def min(a: Tensor, b: Tensor) -> Tensor:
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return Tensor(data=np.minimum(a.data, b.data), _children=(a, b), _op='functional_min')
    elif isinstance(a, Tensor) and isinstance(b, int):
        b_tensor = Tensor(data=np.full(a.shape, b), _children=(Tensor(b), ), _op='cloning_b')
        return Tensor(data=np.minimum(a.data, b_tensor.data), _children=(a, b), _op='functional_min')

def sigmoid(a: Tensor):
    '''
    Simple sigmoid: 1/1+(e^-x)
    '''
    def _backward():
        # backward impl
        pass
    if isinstance(a, Tensor):
        sigmoid_term = 1 / (1 + np.exp(-a.data))
        sigmoid_tensor = Tensor(data=sigmoid_term, _children=(a, ), _op='sigmoid')

        ## Just update here
        sigmoid_tensor._backward = _backward

    else:
        raise TypeError("Its not a tensor type, plese convert it to Tensor using a = Tensor(data=<your_tensor_data>)")

def softmax(a: Tensor, inplace=False) -> Tensor:
    '''
    Standard softmax impl: e ^ (z) / sigma (e^z)
    '''
    def _backward():
        # backward impl
        pass
    
    if isinstance(a, Tensor):
    
        exponential_vector = a.exp()
        sum_vector = exponential_vector.sum()
        softmax_output = exponential_vector.data / sum_vector.data
        
        return Tensor(
            data=softmax_output, _children=(sum_vector, exponential_vector), _op='softmax'
        )
    else:
        raise TypeError("Its not a tensor type")

def online_softmax(a: Tensor):
    def _backward():
        # backward impl
        pass
    pass

def relu(a: Tensor):
    def _backward():
        # backward impl
        pass
    
    if isinstance(a, Tensor):
        return Tensor(max(0, a).data, _children=(a, ), _op='relu')
    else:
        raise TypeError("Its not a tensor type, plese convert it to Tensor using a = Tensor(data=<your_tensor_data>)")

def leaky_relu(a: Tensor, ns=0.01):
    def _backward():
        # backward impl
        pass
    if isinstance(a, Tensor):
        relu_term: Tensor = relu(a)
        leaky_term = min(0,a) * ns
        return Tensor(data=relu_term + leaky_term, _children=(relu_term, leaky_term), _op='leaky_relu')
    else:
        raise TypeError("Its not a tensor type, plese convert it to Tensor using a = Tensor(data=<your_tensor_data>)")

