################## Importing packages ####################

from astropy.io import fits
import numpy as np
import configparser

################## Functions ####################

def combine_darks( config, logfile = None ):
    """
    Combines raw dark frame files (with file numbers ranging from raw_dark_startno and raw_dark_endno in the
    config file) into a single mean dark frame. 
    
    Required Parameters
    -------------------
    
            config          String
            
                                The file name(s) (with paths) of the configuration file.
    
    Optional Parameters
    -------------------
    
            logfile         String or None
                            
                                [ Default = None ]
                            
                                File name (and path) of a log file in which to provide feedback on the 
                                function's progress. If not provided, progress will be printed to the 
                                terminal.
                            
    Config File Parameters Used
    ---------------------------
    
            [COMPUTING]     save_mem
                            max_frames_inmem
        
            [CALIB]         raw_cals_path
                            calib_outpath
                            raw_dark_startno
                            raw_dark_endno
                            
    Output Files Generated
    ----------------------
    
        [calib_outpath]/dark_meanframe_[raw_dark_startno]_[raw_dark_endno].fits
        
                            Fits file containing (in extension 0) the calculated mean dark frame. 
                            
                            Copies some info from original fits file headers to the extension 0 header of this
                            file, as well as saving the start and end file numbers and the total number of
                            files used.
    """
    
    # 
    