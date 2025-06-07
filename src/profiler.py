import cProfile, pstats, io
from pstats import SortKey
import functools
from src.modes import SET_PROFILE_MODE

'''
A profiler is used to list down all the underlying function calls that are executed in the stack trace during a programs execution.
The profiler section of picograd consists of two methods: 
1. The PicoProfiler itself that can be used as a context manager like torch.autograd.profile()
2. profile_op, that can be used as an decorator on top of operations to profile their working. 

The 2nd approach can be used to profile custom operators that get dispatched as cuda/triton kernels
The first approach can be commonly used to profile neural nets that are built using picoGrad. 

'''


if SET_PROFILE_MODE == True:
    print("------------------------PROFILER MODE ENABLED------------------------")
    print("To disable it, set SET_PFOILE_MODE=False in modes.py")
else:
    print("Profiler has been disabled, to enable it please go to modes and set SET_PROFILE_MODE=True")

def profile_op(func):
    '''
    '''
    ## Wrapper function for the core op
    def wrapper(*args, **kwargs):
        
        profiler = cProfile.Profile()
        profiler.enable()

        results = func(*args, **kwargs)

        profiler.disable()
        
        ## This needs to be conditional, because I dont want to
        ## dump everything I find. 
        profiler.dump_stats('profile')   
        profiler.print_stats()

        return results
    
    ## The profile op function will return the normal results 
    ## It depends on the SET_PROFILE_MODE in modes.py
    def dont_wrap(*args, **kwargs):
        results = func(*args, **kwargs)
        return results

    if SET_PROFILE_MODE:
        return wrapper

    return dont_wrap

## vizing profile logs, todo
def vizualise_profile_logs(logs_file): #todo
    pass
            
## profiler class, honestly profilers needs to be its own
## subthing, not in a single file, it feels jank. 
class PicoProfiler:
    ''' 
        This class implements a context manager similar to torch.autograd.profile()
        (dont quote me on the api).

        Usage: 
        
        with PicoProfiler() as prof:
            a = pico.Tensor([2.0, 3.4], label='a')
            b = pico.Tensor([3.5, 3.5], label='b')
            c = a + b
    '''
    
    def __init__(self):
        self.profiler = cProfile.Profile()

    def __enter__(self):
        self.profiler.enable()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.profiler:
            self.profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s)
            ps.print_stats(10)  # Only top 10 entries
            print(s.getvalue()) 





