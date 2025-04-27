## PicoTensor: PicoGrad v0
import time
import numpy as np

from src.profiler import profile_op

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

    def _check_shapes(self, other):
        return self.shape == other.shape
    
    ## Broadcasting utils - have to implement this on my own
    def broadcast(self, a, b):
        if isinstance(a, Tensor) and isinstance(b, Tensor):
            a_b, b_b = np.broadcast_arrays(a, b)
            return Tensor( data=a_b ), Tensor( data=b_b )

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


    @profile_op
    def __add__(self, other):
        assert type(other) == Tensor, "Other is a tensor, op=add"
        assert other.dtype == self.dtype, f"Other and me are of same dtype: {self.dtype}"

        try:
            if self._check_shapes(other) == True:
                out = Tensor(data = self.data + other.data, _children=(self, other), _op="+")
            else:
                ## Todo: add the backward flow for this
                a_b, b_b = self.broadcast(self.data, other.data)
                out = Tensor(data = a_b.data + b_b.data, _op='+')
        except Exception as e:
            raise e

        ## Backward function 
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward

        return out

    ## will come back to this 
    def __sub__(self, other):
        assert isinstance(other, Tensor), "Other is a tensor op=sub"
        assert other.dtype == self.dtype, f"Other and me dont have the same type {self.dtype} != {other.dtype}"
        try:
            if self._check_shapes(other) == True:
                out = Tensor(data = self.data - other.data, _children = (self, other), _op='-')
            else:
                a_b, b_b = self.broadcast(self.data, other.data)
                out = Tensor(data = a_b.data + b_b.data, _op = "-")

        except Exception as e:
            raise e
        
        return out

    
    def __mul__(self, other):
        assert isinstance(other, Tensor), "Other is a tensor, op=mul"
        assert other.dtype == self.dtype, f"Other and me dont have the same type (self.dtype} != {other.dtype}"

        try:
            if self._check_shapes(other) == True:
                out = Tensor(data=self.data * other.data, _children=(self, other), _op="*")
            else:
                a_b, b_b = self.broadcast(self.data, other.data)
                out = Tensor(data = a_b.data * b_b.data, _children = (a_b, b_b), _op='-')

        except Exception as e:
            raise e

        def _backward():
            self.grad *= out.grad
            other.grad *= out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        out = Tensor(data = self.data  / other.data, _children=(self, other), _op='/')

        def _backward():
            self.grad /= out.grad
            other.grad /= out.grad

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


    ## Basic math ops section: For now all of these are numpy
    ## Todo: tanh, sin, cosine
    def log(self):
        log_result = np.log(self.data)

        out = Tensor(data=log_result, _children=(self,), _op='log')
        return out

    def exp(self):
        exp_result = np.exp(self.data)

        out = Tensor(data=exp_result, _children=(self,), _op='exp')
        return out
    
    def sum(self):
        sum_result = np.sum(self.data)

        out = Tensor(data=sum_result, _children=(self,), _op='sum')
        return out

    # backward function for computing 
    # gradients
    def backward(self):
        print("Visiting this function right now?")

        topo = []
        visited = set()
        visited_ops = set()
        
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
