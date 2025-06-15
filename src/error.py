from pico import Tensor

class InputException(Exception):
    def __init__(self, input_string, options, ):
        super().__init__()
        self.input = input_string
        self.options = options
    
    def __str__(self):
        return f'Input {self.input} doesnt belong to options - {self.options}'

class NotATensorException(Exception):
    pass
