import logging
from src.modes import SET_LOGGING_MODE
import functools

if SET_LOGGING_MODE == True:
    print("Logger enabled")
    print("To disable it, please go to modes.py")
else:
    print("Logger has been disabled, please enable in modes.py")

def init_logger_setup(filename):
    pass

def log_this_op(func, disable=True):
    def wrapper(args, kwargs):
        pass

    def dont_wrap():
        pass


