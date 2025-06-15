import numpy as np

from pico import Tensor
from error import InputException


def max(a: Tensor, b: Tensor) -> Tensor:
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return Tensor(data=np.maximum(a.data, b.data), _children=(a, b), _op='functional_max')
    
    elif isinstance(a, Tensor) and isinstance(b, int):
        b_tensor = Tensor(data=np.full(a.shape, b), _children=(Tensor(b), ), _op='cloning_b')
        return Tensor(data=np.maximum(a.data, b_tensor.data), _children=(a, b), _op='functional_max')

    else:
        return TypeError("inputs must be either (Tensor, int) or (Tensor, Tensor)")
    
def min(a: Tensor, b: Tensor) -> Tensor:
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return Tensor(data=np.minimum(a.data, b.data), _children=(a, b), _op='functional_min')
    
    elif isinstance(a, Tensor) and isinstance(b, int):
        b_tensor = Tensor(data=np.full(a.shape, b), _children=(Tensor(b), ), _op='cloning_b')
        return Tensor(data=np.minimum(a.data, b_tensor.data), _children=(a, b), _op='functional_min')

    else:
        return TypeError("inputs must be either (Tensor, int) or (Tensor, Tensor)")


def sigmoid(a: Tensor):
    '''
    Simple sigmoid: 1/1+(e^-x)
    '''

    if isinstance(a, Tensor):
        sigmoid_term = 1 / (1 + np.exp(-a.data))
        sigmoid_tensor = Tensor(data=sigmoid_term, _children=(a, ), _op='sigmoid')

        def _backward():
            # backward impl
            a.grad += sigmoid_tensor.grad * (sigmoid_tensor.data * (1 - sigmoid_tensor.data))

        ## Just update here
        sigmoid_tensor._backward = _backward
        return sigmoid_tensor
    
    else:
        raise TypeError("Its not a tensor type, plese convert it to Tensor using a = Tensor(data=<your_tensor_data>)")

def softmax(a: Tensor, inplace=False, mode='normal') -> Tensor:
    '''
    Standard softmax impl: e ^ (z) / sigma (e^z)
    '''
    def _backward():
        # backward impl
        if mode == 'normal':
            pass

        if mode == 'safe':
            pass
    
    if mode == 'normal':
        if isinstance(a, Tensor):
        
            exponential_vector = a.exp()
            sum_vector = exponential_vector.sum()
            softmax_output = exponential_vector.data / sum_vector.data
            
            return Tensor(
                data=softmax_output, _children=(sum_vector, exponential_vector), _op='softmax'
            )
        else:
            raise TypeError("Its not a tensor type")
    
    elif mode == 'safe':
        if isinstance(a, Tensor):

            max_term = a.max()
            shifted = a - max_term  # Subtract max before exp
            exponential_vector = shifted.exp()
            sum_vector = exponential_vector.sum()
            softmax_output = Tensor(
                data=exponential_vector.data / sum_vector.data, _children=(sum_vector, exponential_vector), _op='softmax'
            )

            softmax_output.backward = _backward
            return softmax_output

        else:
            raise TypeError("Its not a tensor type")

    else:
        return InputException(mode, ['online', 'safe', 'normal'])


def relu(a: Tensor):
    
    if isinstance(a, Tensor):
        relu_term = Tensor(max(0, a).data, _children=(a, ), _op='relu')
        def _backward():
        # backward impl
            a.grad += relu_term.grad * (a.data > 0)
        
        relu_term.backward = _backward
        
        return relu_term
    
    else:
        raise TypeError("Its not a tensor type, plese convert it to Tensor using a = Tensor(data=<your_tensor_data>)")

def leaky_relu(a: Tensor, ns=0.01):

    if isinstance(a, Tensor):
        relu_term: Tensor = relu(a)
        leaky_term = min(0,a) * ns

        leaky_relu = Tensor(data=relu_term + leaky_term, _children=(relu_term, leaky_term), _op='leaky_relu')

        def _backward():
            a.grad += leaky_relu.grad * np.where(a.data > 0, 1.0, ns)

        leaky_relu.backward = _backward
        return leaky_relu
    else:
        raise TypeError("Its not a tensor type, plese convert it to Tensor using a = Tensor(data=<your_tensor_data>)")


def swish(a: Tensor, beta: int):

    if isinstance(a, Tensor):
        beta_sigmoid_term = 1 / (1 + np.exp(beta * -a.data))
        swish_output = Tensor(data=beta_sigmoid_term, _children=(beta_sigmoid_term, ), _op='swish')

        def _backward():
            pass

        swish_output.backward = _backward
        return swish_output

    else:
        raise TypeError("Its not a tensor type, plese convert it to Tensor using a = Tensor(data=<your_tensor_data>)")
    

