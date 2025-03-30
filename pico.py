## PicoTensor: PicoGrad v0

import time
import numpy as np
import cProfile
import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', label='', dtype=''):
        ## This if for CPU
        self.data = np.array(data)
        self.grad = 0.0
        self.label = label
        self.device = None
        
        self._backward = lambda: None
        self._op = _op
        self._prev = set(_children)

    def T(self):
        pass

    def __add__(self, other):

        assert (type(other) == Tensor, "Other is a tensor, op=add")        
        ## apparently we set the children here? 
        out = Tensor(data = self.data + other.data, _children=(self, other), _op="+")

        ## Backward function 
        def _backward():
            self.grad += 1.0 * other.grad
            other.grad += 1.0 * other.grad
        
        out._backward = _backward

        return out
   
   ## will come back to this 
    def __sub__(self, other):
        assert isinstance(other, Tensor), "Other is a tensor op=sub"
        out = Tensor(data = self.data + -other.data)
        print(out)

        return out

    
    def __mul__(self, other):
        assert isinstance(other, Tensor), "Other is a tensor, op=mul"
        out = Tensor(data=self.data * other.data, _children=(self, other), _op="*")
        def _backward():
            self.grad *= out.grad
            other.grad *= out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        assert (type(other) == Tensor, "Check whether object is tensor or not") 

        return self + other
    
    def __sub__(self, other):
        pass

    ## Indexing and slicing for tensor object
    def __getitem__(self, indices=None):
        if indices is None:
            return self.data

        if isinstance(indices, (int, slice, tuple)):
            return self.data[indices]

        raise TypeError(f"Invalid index type: {type(indices)}")

    # backward function for computing 
    # gradients
    def backward(self):
        print("Visiting this function right now?")

        topo = []
        visited = set()
        
        ## What does this do? Ik it constructs the backward graph
        ## But how does it do it? what is topo sort? 
        def build_topo_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo_sort(child)
                
                topo.append(v)

        build_topo_sort(self)
        
        print(f"Topological graph for {self}: {topo}")
        return topo

    def __repr__(self):    
        return str(self.data)

def tensor():
    pass

