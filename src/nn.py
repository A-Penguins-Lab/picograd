from pico import Tensor

class PicoModule:
    def __init__(self):
        pass

    def forward(self):
        return NotImplemented("Please implement this as you like :)")

class Linear(PicoModule):
    def __init__(self):
        super().__init__()
    
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)
        
