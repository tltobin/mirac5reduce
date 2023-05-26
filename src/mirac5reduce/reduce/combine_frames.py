################## Importing packages ####################

import os
import numpy as np
import configparser

from ..utils.utils import get_raw_filenames, write_mean_frame
from ..utils.calcframes import calc_mean_frame

################## Functions ####################

def meanframe( config, frametype, 
               datapath = None, startno = None, endno = None, outfile = None,
               logfile = None, debug = False ):
    """
    Combines raw frame files (with file numbers ranging from startno and endno) into a single mean frame and
    saves the result to a fits file. 
    
    Note: This does assume that all raw frames are stored in their own fits files, under the extension
    indicated by data_ext in the config file.
    
    Required Parameters
    -------------------
    
            config          String
            
                                The file name(s) (with paths) of the configuration file.
            
            frametype       String: 'dark', 'flat', 'obs'
                                
                                Image type (i.e. dark, flat, obs) of the data frames averaged.
                                
                                This determines the default values for the following optional parameters
                                unless they are specified manually:
                                    - datapath
                                    - startno 
                                    - endno  
                                    - outfile 
                                See the descriptions for these parameters below for more info.
            
    Optional Parameters
    -------------------
            
            datapath        String or None
                                
                                [ Default = None ]
                                
                                The path where the raw fits file frames to be averaged are stored.
                                
                                If set to None, will use value imported from the provided config file as
                                determined by the specified frametype.
                                
                                If frametype is 'dark' or 'flat', uses config value, raw_cals_path.
                                
                                Otherwise, uses config value, raw_data_path.
            
            startno         Integer or None
                                
                                [ Default = None ]
                                
                                File number of the first file in the range of files to be averaged together.
                                
                                If set to None, will use value imported from the provided config file as
                                determined by the specified frametype.
                                
                                If frametype is 'dark', uses config value, raw_dark_startno.
                                
                                If frametype is 'flat', uses config value, raw_flat_startno.
                                
                                Otherwise, uses config value, raw_data_startno.
            
            endno           Integer or None
                                
                                [ Default = None ]
                                
                                File number of the last file in the range of files to be averaged together.
                                
                                If set to None, will use value imported from the provided config file as
                                determined by the specified frametype.
                                
                                If frametype is 'dark', uses config value, raw_dark_endno.
                                
                                If frametype is 'flat', uses config value, raw_flat_endno.
                                
                                Otherwise, uses config value, raw_data_endno.
            
            outfile         String or None
                                
                                [ Default = None ]
                                
                                Path and name to which the fits file with the resulting mean frame will be 
                                saved. 
                                
                                If set to None, will be determined using values imported from the provided 
                                config file as determined by the specified frametype.
                                
                                If frametype is 'dark' or 'flat', uses config value, calib_outpath, to set 
                                outfile = '[calib_outpath]/[frametype]_[startno]_[endno].fits'.
                                
                                Otherwise, uses config value, reduce_outpath, to set 
                                outfile = '[reduce_outpath]/[frametype]_[startno]_[endno].fits'.
            
            logfile         String or None
                            
                                [ Default = None ]
                            
                                File name (and path) of a log file in which to provide feedback on the 
                                function's progress. If not provided, progress will be printed to the 
                                terminal.
    
            debug          Boolean
                            
                                [ Default = False ]
                            
                                If set to True, will print additional debugging information to the terminal.
                            
    Config File Parameters Used
    ---------------------------
    
        Always used:
    
            [COMPUTING]     save_mem
                            max_frames_inmem
                            
            [DATA_ARCH]     data_ext
                            raw_name_fmt
        
        Sometimes used (see Optional Parameters above for more info):
            
            [REDUCTION]     raw_data_path
                            reduce_outpath
                            raw_data_startno
                            raw_data_endno
            
            [CALIB]         raw_cals_path
                            calib_outpath
                            raw_dark_startno
                            raw_dark_endno
                            raw_flat_startno
                            raw_flat_endno
                            
    Output Files Generated
    ----------------------
    
        [outfile]
        
                            Fits file containing (in extension 0) the calculated mean frame. 
                            
                            Copies some info from original fits file headers to the extension 0 header of this
                            file, as well as saving the start and end file numbers and the total number of
                            files used.
    """
    
    # Retrieves config file
    conf = configparser.ConfigParser()
    _ = conf.read(config)
    
    # imports and saves all needed config file values, converting them to the applicable data type
    save_mem         = conf['COMPUTING'].getboolean('save_mem')
    max_frames_inmem = conf['COMPUTING']['max_frames_inmem']
    data_ext         = conf['DATA_ARCH']['data_ext']
    raw_name_fmt     = conf['DATA_ARCH']['raw_name_fmt']
    try:
        max_frames_inmem = int(max_frames_inmem)
    except:
        max_frames_inmem = None
    try:
        data_ext = int(data_ext)
    except:
        data_ext = None
    
    # Parses optional keys that may have been provided to override the config values and sets any default
    #   values based on the provided frametype
    frametype = frametype.lower()
    
    if frametype in ['dark','flat']:
        if datapath is None:
            datapath = conf[  'CALIB'  ]['raw_cals_path']
        if startno is None:
            startno  = conf[  'CALIB'  ].getint('raw_{0}_startno'.format(frametype))
        if endno is None:
            endno    = conf[  'CALIB'  ].getint('raw_{0}_endno'.format(frametype))
        if outfile is None:
            outpath  = conf[  'CALIB'  ]['calib_outpath']
            outfile  = os.path.join( outpath, '{0}_{1}_{2}.fits'.format(frametype, startno, endno) )
    
    else:
        if datapath is None:
            datapath = conf['REDUCTION']['raw_data_path']
        if startno is None:
            startno  = conf['REDUCTION'].getint('raw_data_startno')
        if endno is None:
            endno    = conf['REDUCTION'].getint('raw_data_endno')
        if outfile is None:
            outpath  = conf['REDUCTION']['reduce_outpath']
            outfile  = os.path.join( outpath, '{0}_{1}_{2}.fits'.format(frametype, startno, endno) )
        
    
    
    # Debugging message checkpoint
    if debug:
        feedbacklines = ['MEANFRAME.DEBUG          Provided frametype: {0}'.format(frametype),
                         'MEANFRAME.DEBUG          Parameters set manually or determined from frametype:',
                         'MEANFRAME.DEBUG              {0: >16} : {1}'.format( 'datapath', datapath ),
                         'MEANFRAME.DEBUG              {0: >16} : {1}'.format( 'startno', startno ),
                         'MEANFRAME.DEBUG              {0: >16} : {1}'.format( 'endno', endno ),
                         'MEANFRAME.DEBUG              {0: >16} : {1}'.format( 'outfile', outfile ),
                         'MEANFRAME.DEBUG          Parameters retrieved from config file:',
                         'MEANFRAME.DEBUG              {0: >16} : {1}'.format( 'save_mem', save_mem ),
                         'MEANFRAME.DEBUG              {0: >16} : {1} -> {2}'.format( 
                                'max_frames_inmem', conf['COMPUTING']['max_frames_inmem'], max_frames_inmem ),
                         'MEANFRAME.DEBUG              {0: >16} : {1} -> {2}'.format( 
                                'data_ext', conf['DATA_ARCH']['data_ext'], data_ext ),
                         'MEANFRAME.DEBUG              {0: >16} : {1}'.format( 'raw_name_fmt', raw_name_fmt ) ]
        for flin in feedbacklines:
            print(flin)
    
    # Writes quick note to logfile or terminal
    feedback_msg = 'MEANFRAME:           Retrieving list of files with numbers {0}-{1}.'.format( startno, endno )
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    # Retrieves the list of file names for the requested file numbers
    filelist = get_raw_filenames( raw_name_fmt, startno, endno, datapath  )
    
    
    
    # Debugging message checkpoint
    if debug:
        print('MEANFRAME.DEBUG          File names retrieved: {0}'.format(len(filelist)))
    
    # Writes quick note to logfile or terminal regarding whether memory saving is turned on or not for this
    feedback_msg = 'MEANFRAME:           Calculating mean {0} with save_mem = {1}'.format( frametype, str(save_mem) )
    if save_mem: 
        feedback_msg += ' (max {0} frames)'.format( max_frames_inmem )
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    
    # Splits here to calculate average frame from data files in memory saving mode or directly
    if save_mem:
        avgframe = calc_mean_frame( [ os.path.join( datapath, fname ) for fname in filelist ], 
                                  ext = data_ext, maxframes = max_frames_inmem, logfile = logfile )
    else:
        avgframe = calc_mean_frame( [ os.path.join( datapath, fname ) for fname in filelist ], 
                                  ext = data_ext, maxframes = None, logfile = logfile )
    

    
    # Writes quick note to logfile or terminal
    feedback_msg = 'MEANFRAME:           Saving Results to {0}.'.format( outfile )
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    
    # Uses write_mean_frame function to save the calculated data to the desired fits file
    write_mean_frame( outfile, avgframe, frametype, filelist, raw_filepath = datapath )
    
    
    
    
    