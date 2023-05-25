################## Importing packages ####################

from astropy.io import fits
import numpy as np
import configparser

################## Functions ####################

def combine_darks( config, combine_as = 'mean', logfile = None, **kwargs ):
    """
    Combines raw dark frame files (with file numbers ranging from raw_dark_startno and raw_dark_endno in the
    config file) into a single mean or median dark frame. 
    """
    
    