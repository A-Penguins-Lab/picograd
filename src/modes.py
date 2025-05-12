'''
This file is for setting the modes. Currently it has three modes which are: 
    1. Profiler mode: that enables profile_op in profiler.py
    2. Debug mode that spits out logs during ops 
    3. Logger mode: not implemented yet as of 27th april

These modes have to be set as global variables or the user should be given an option to override them as they desire. 
'''

## env file
SET_PROFILE_MODE: bool = False
SET_DEBUG_MODE: bool = True
