import cProfile, pstats, io
from pstats import SortKey
import functools
from src.modes import SET_PROFILE_MODE

if SET_PROFILE_MODE == True:
    print("------------------------PROFILER MODE ENABLED------------------------")
    print("To disable it, set SET_PFOILE_MODE=False in modes.py")

def profile_op(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        results = func(*args, **kwargs)
        profiler.disable()
        
        profiler.print_stats()
        return results

    def dont_wrap(*args, **kwargs):
        results = func(*args, **kwargs)
        return results

    if SET_PROFILE_MODE:
        return wrapper
    
    return dont_wrap

def dump_profile_logs(logs):
    pass

def vizualise_profile_logs(logs_file):
    pass

            
class PicoProfiler:
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