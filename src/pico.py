## PicoTensor: PicoGrad v0

import time
import numpy as np
import cProfile
import logging
import numpy as np

from broadcast_utils import check_shapes, broadcast

class Tensor:
    def __init__(self, data, _children=(), _op='', label='', dtype=np.float64):
        ## This if for CPU
        
        self.dtype = dtype
        self.data = np.array(data, dtype=self.dtype)
        self.grad = 0.0
        self.label = label
        self.device = None
        self._op = _op
        self._prev = set(_children)
    
        self._properties()

    def _properties(self):
        self._backward = lambda: None
        self.dtype = self.data.dtype
        self.shape = self.data.shape

    def transpose(self):
        if len(self.shape) >= 2:
            print(self.data)

    def copy(self):
        print("Don't copy you donkey")
        return Tensor( data = self.data ) 

    def not_view(self, dims):
        if isinstance(dims, tuple):
            res = np.reshape(self.data, dims)
            return Tensor( data = res )

    def flatten(self):
        res = self.not_view( dims = (-1,1) )
        return Tensor( data = res )

    def __add__(self, other):

        assert type(other) == Tensor, "Other is a tensor, op=add"
        assert other.dtype == self.dtype, f"Other and me are of same dtype: {self.dtype}"

        if check_shapes(self, other) == True:
            out = Tensor(data = self.data + other.data, _children=(self, other), _op="+")
        else:
            ## Todo: add the backward flow for this
            a_b, b_b = broadcast(self.data, other.data)
            out = Tensor(data = a_b.data + b_b.data, _op='+')

        ## Backward function 
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward

        return out

    ## will come back to this 
    def __sub__(self, other):
        assert isinstance(other, Tensor), "Other is a tensor op=sub"
        out = Tensor(data = self.data + -other.data)
        
        return out

    
    def __mul__(self, other):
        assert isinstance(other, Tensor), "Other is a tensor, op=mul"
        out = Tensor(data=self.data * other.data, _children=(self, other), _op="*")
        def _backward():
            self.grad *= out.grad
            other.grad *= out.grad

        out._backward = _backward
        return out


    def __radd__(self, other):
        assert type(other) == Tensor, "Check whether object is tensor or not"
        return self + other
    
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
    
