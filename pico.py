import time
import numpy as np
import cProfile
import numpy as np


class Tensor:
    def __init__(self, data, label='', _op='', _children=()):
        
        ## This if for CPU
        self.data = np.array(data)
        self.grad = 0.0
        self.label = label
        
        self._backward = lambda: None
        self._op = _op
        self._prev = set(_children)

    def T(self):
        pass

    def __add__(self, other):
        # assert (type(other) == Tensor, "Other is a tensor, op=add")
        out = Tensor(self.data + other.data)

        return out
    
    def __sub__(self, other):
        # assert (type(other) == Tensor, "Other is a tensor op=sub")
        out = Tensor(self.data - other.data)

        return out
    
    def __mul__(self, other):
        # assert (type(other) == Tensor, "Other is a tensor, op=mul")
        out = Tensor(self.data * other.data)

        return out

    ## Indexing and slicing for tensor object
    def __getitem__(self, indices=None):
        if indices is None:
            return self.data

        if isinstance(indices, (int, slice, tuple)):
            return self.data[indices]

        raise TypeError(f"Invalid index type: {type(indices)}")

    def backward(self):
        pass

    def __repr__(self):
        return str(self.data)

def tensor():
    pass

if __name__ == "__main__":
    start_time = time.time()
    a = Tensor([2.0, 3.4], label='a')
    b = Tensor([3.5, 3.5], label='b')
    c = a + b

    end_time = time.time()

    ## This needs some speeding up
    print(end_time - start_time)
    print(c[1])
    print(c, 'data: ', c.data, 'data type: ', type(c.data))
