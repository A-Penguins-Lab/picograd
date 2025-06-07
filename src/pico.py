## PicoTensor: PicoGrad v0
import numpy as np

from src.profiler import profile_op
from logger_setup import setup_logging
from modes import SET_LOGGING_MODE

pico_logger = setup_logging('pico.log')

class Tensor:
    def __init__(self, data, _children=(), _op='', label='', dtype=np.float64):
        '''
        The core tensor class, this class is the center of picograd that consists of 
        all the operations and the backward engine. The arguments for this class include: 

        self.dtype: str -> The datatype of the tensor that is adapted from numpy.dtypes. Please refer to their documentation
        self.data: np.ndarray -> The actual tensor, for version-0 picoTensor is a numpy array
        self.grad: float -> The gradient of the tensor, calculated during the backward pass
        self.label: str -> A string denoting the name of the tensor used. Mainly used during backward graphs computation. 
        self.device: str -> Options would be cpu/cuda for cpu/gpu acccess
        self._op -> Name of the operation used during computation
        self._prev -> When computing a new tensor, an operation must be applied on one or more tensors, this set stores the previous 
        tensor objects that were used for graph creation. 

        Usage:
        1. For scalar varaibles: 
            a = pico.Tensor(1.0, label='a')
        2. For vector variables: 
            a = pico.Tensor([1.3,5,6.2], label='a_vector')

        The current features that the pico.Tensor class has are: 
            1. A backward() function that can construct the backward graph.
            2. Brodcasting adapted from numpy
            3. Operations: add, subtract, multiply, divide, log, sum, 
            4. Indexing and slicing. 
        '''

        ## This if for CPU
        
        self.dtype = dtype
        self.data = np.array(data, dtype=self.dtype)
        self.grad = np.zeros_like(self.data)
        self.label = label
        self._op = _op
        self._prev = set(_children)
        self._set_tensor_properties()

        pico_logger.info(f"[INIT]: Tensor created with label={label}, shape={self.shape}, dtype={self.dtype}")


    def _set_tensor_properties(self):
        self._backward = lambda: None
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.device = None

    def _check_shapes(self, other):
        return self.shape == other.shape
    
    ## Broadcasting utils - have to implement this on my own
    def broadcast(self, a, b):
        if isinstance(a, Tensor) and isinstance(b, Tensor):
            a_b, b_b = np.broadcast_arrays(a, b)
            return Tensor( data=a_b ), Tensor( data=b_b )

    def max(self):
        if len(self.data) > 1:
            return max(self.data)
        else:
            return self.data[0]
        

    def T(self):
        if len(self.shape) >= 2:
            print(self.data)

    def copy(self):

        return Tensor( data = self.data )

    def not_view(self, dims):
        if isinstance(dims, tuple):
            res = np.reshape(self.data, dims)
            return Tensor( data = res )

    def flatten(self):
        res = self.not_view( dims = (-1,1) )
        return Tensor( data = res )

    def __add__(self, other):
        assert isinstance(other, Tensor), pico_logger.error("Expected 'other' to be of type Tensor for add operation.")
        assert other.dtype == self.dtype, pico_logger.error(f"Data type mismatch: self.dtype={self.dtype}, other.dtype={other.dtype}")

        try:
            if self._check_shapes(other) == True:
                out = Tensor(data = self.data + other.data, _children=(self, other), _op="+")
                pico_logger.debug(f"[OP: +]: Created Tensor -> shape={out.shape}, dtype={out.dtype}")
            else:
                ## Todo: add the backward flow for this
                a_b, b_b = self.broadcast(self.data, other.data)
                out = Tensor(data = a_b.data + b_b.data, _op='+')
                pico_logger.debug(f"[OP: + (broadcasted)]: Created Tensor -> shape={out.shape}, dtype={out.dtype}")

        except Exception as e:
            pico_logger.error(f"[ERROR in __mul__]: Shape mismatch or invalid operation. {e}")

        ## Backward function 
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward

        return out

    ## will come back to this 
    def __sub__(self, other):
        assert isinstance(other, Tensor), pico_logger.error("Expected 'other' to be of type Tensor for add operation.")
        assert other.dtype == self.dtype, f"Other and me dont have the same type {self.dtype} != {other.dtype}"
        try:
            if self._check_shapes(other) == True:
                out = Tensor(data = self.data - other.data, _children = (self, other), _op='-')
            else:
                a_b, b_b = self.broadcast(self.data, other.data)
                out = Tensor(data = a_b.data + b_b.data, _op = "-")

        except Exception as e:
            pico_logger.error(f"[ERROR in __mul__]: Shape mismatch or invalid operation. {e}")
        
        return out

    
    def __mul__(self, other):
        assert isinstance(other, Tensor), pico_logger.error("Expected 'other' to be of type Tensor for add operation.")
        assert other.dtype == self.dtype, f"Other and me dont have the same type {self.dtype} != {other.dtype}"

        try:
            if self._check_shapes(other) == True:
                out = Tensor(data=self.data * other.data, _children=(self, other), _op="*")
            else:
                a_b, b_b = self.broadcast(self.data, other.data)
                out = Tensor(data = a_b.data * b_b.data, _children = (a_b, b_b), _op='-')

        except Exception as e:
            pico_logger.error(f"[ERROR in __mul__]: Shape mismatch or invalid operation. {e}")
            raise e
        
        def _backward():
            self.grad += self.data * out.grad
            other.grad += other.data * out.grad

        out._backward = _backward
        return out

    ## Do we do broadcasting to truediv? 
    def __truediv__(self, other):
        assert isinstance(other, Tensor), pico_logger.error("Expected 'other' to be of type Tensor for add operation.")
        assert other.dtype == self.dtype, f"Other and me dont have the same type {self.dtype} != {other.dtype}"

        out = Tensor(data = self.data  / other.data, _children=(self, other), _op='/')

        def _backward():
            self.grad += self.data / out.grad
            other.grad += other.data / out.grad

        out._backward = _backward

        return out

    ## Indexing and slicing for tensor object
    def __getitem__(self, indices=None):
        if indices is None:
            return self.data

        if isinstance(indices, (int, slice, tuple)):
            return self.data[indices]

        raise TypeError(f"Invalid index type: {type(indices)}")

    ## Basic math ops section: For v0 all of these are numpy
    ## Todo: tanh, sin, cosine
    def log(self):
        assert isinstance(self, Tensor), pico_logger.error("Expected 'other' to be of type Tensor for add operation.")
        log_result = np.log(self.data)

        out = Tensor(data=log_result, _children=(self,), _op='log')

        def _backward():
            ## numpy gods allow this, how gracious of them
            self.grad += ( 1 / self.data ) * out.grad

        self._backward = _backward

        return out

    def exp(self):
        assert isinstance(self, Tensor), pico_logger.error("Expected 'other' to be of type Tensor for add operation.")

        exp_result = np.exp(self.data)
        out = Tensor(data=exp_result, _children=(self,), _op='exp')

        def _backward():
            ## omg numpy allows for this as well, i feel so guilty :)
            self.grad += np.exp(self.data) * out.grad

        self._backward = _backward

        return out
    
    def sum(self):
        assert isinstance(self, Tensor), pico_logger.error("Expected 'other' to be of type Tensor for add operation.")
        
        sum_result = np.sum(self.data)
        name = self.label

        def _backward():
            self.grad = np.ones_like(self.data) * out.grad

        self._backward = _backward
        out = Tensor(data=sum_result, _children=(self,), _op=f'{name}_sum')

        return out

    # backward function for computing 
    # gradients
    def backward(self):
        pico_logger.info(f"[BACKWARD]: Starting backward pass for Tensor with label={self.label}")

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
        
        pico_logger.info(f"Topological graph for {self}: {topo}")
        return topo

    def __repr__(self):    
        return str(self.data)