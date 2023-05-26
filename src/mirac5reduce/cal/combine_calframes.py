################## Importing packages ####################

import os
from astropy.io import fits
import numpy as np
import configparser

from ..utils.utils import get_raw_filenames
from ..utils.memory_saver import lm_mean_frame

################## Functions ####################

def combine_darks( config, logfile = None, debug = False ):
    """
    Combines raw dark frame files (with file numbers ranging from raw_dark_startno and raw_dark_endno in the
    config file) into a single mean dark frame. 
    
    Note: This does assume that all raw dark frames are stored in their own fits files, under the extension
    indicated by data_ext in the config file.
    
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
    
            debug          Boolean
                            
                                [ Default = False ]
                            
                                If set to True, will print additional debugging information to the terminal.
                            
    Config File Parameters Used
    ---------------------------
    
            [COMPUTING]     save_mem
                            max_frames_inmem
        
            [CALIB]         raw_cals_path
                            calib_outpath
                            raw_dark_startno
                            raw_dark_endno
                            
            [DATA_ARCH]     data_ext
                            raw_name_fmt

                            
    Output Files Generated
    ----------------------
    
        [calib_outpath]/dark_[raw_dark_startno]_[raw_dark_endno].fits
        
                            Fits file containing (in extension 0) the calculated mean dark frame. 
                            
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
    raw_cals_path    = conf[  'CALIB'  ]['raw_cals_path']
    calib_outpath    = conf[  'CALIB'  ]['calib_outpath']
    raw_dark_startno = conf[  'CALIB'  ].getint('raw_dark_startno')
    raw_dark_endno   = conf[  'CALIB'  ].getint('raw_dark_endno')
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
    
    
    # Debugging message checkpoint
    if debug:
        print('    COMBINE_DARKS.DEBUG      Parameters retrieved from config file:')
        for parname, par in zip( ['save_mem','raw_cals_path','calib_outpath','raw_dark_startno','raw_dark_endno','raw_name_fmt'], 
                                 [ save_mem , raw_cals_path , calib_outpath , raw_dark_startno , raw_dark_endno , raw_name_fmt ]):
            print('    COMBINE_DARKS.DEBUG          {0: >16} : {1}'.format( parname, par ) )
        print('    COMBINE_DARKS.DEBUG          {0: >16} : {1}'.format( parname, par ) )
        print('    COMBINE_DARKS.DEBUG          {0: >16} : {1} -> {2}'.format( 
                                'max_frames_inmem', conf['COMPUTING']['max_frames_inmem'], max_frames_inmem ))
        print('    COMBINE_DARKS.DEBUG          {0: >16} : {1} -> {2}'.format( 
                                'data_ext', conf['DATA_ARCH']['data_ext'], data_ext ))
    
    # Writes quick note to logfile or terminal
    feedback_msg = 'COMBINE_DARKS:       Retrieving list of files with numbers {0}-{1}.'.format( raw_dark_startno, raw_dark_endno )
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    # Retrieves the list of file names for the requested file numbers
    filelist = get_raw_filenames( raw_name_fmt, raw_dark_startno, raw_dark_endno, raw_cals_path  )
    
    
    
    # Debugging message checkpoint
    if debug:
        print('    COMBINE_DARKS.DEBUG      File names retrieved: {0}'.format(len(filelist)))
    
    # Writes quick note to logfile or terminal regarding whether memory saving is turned on or not for this
    feedback_msg = 'COMBINE_DARKS:       Calculating mean dark with save_mem = {0}'.format( str(save_mem) )
    if save_mem: 
        feedback_msg += ' (max {0} frames)'.format( max_frames_inmem )
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    
    # Splits here to calculate average frame from data files in memory saving mode or directly
    if save_mem:
        avgframe = lm_mean_frame( [ os.path.join( raw_cals_path, fname ) for fname in filelist ], 
                                  ext = data_ext, maxframes = max_frames_inmem, logfile = logfile )
    else:
        avgframe = lm_mean_frame( [ os.path.join( raw_cals_path, fname ) for fname in filelist ], 
                                  ext = data_ext, maxframes = None, logfile = logfile )
    
    
    # Generates name of fits file results will be saved to
    outfile_name = 'dark_{0}_{1}.fits'.format( raw_dark_startno, raw_dark_endno )
    
    # Writes quick note to logfile or terminal
    feedback_msg = 'COMBINE_DARKS:       Saving Results to {0}.'.format( os.path.join( calib_outpath, outfile_name ) )
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
        
    
    # Creates hdu to save to file with mean frame
    hdu = fits.PrimaryHDU( avgframe )
    
    # Populates header directly with general info 
    hdu.header['FILETYPE'] =   'Combined Dark'
    hdu.header['NFRAMES' ] = ( len(filelist), 'Number raw frames used' )
    hdu.header['FILE_STR'] = ( filelist[0]  , 'First raw file used' )
    hdu.header['FILE_END'] = ( filelist[-1] , 'Last raw file used' )
    hdu.header['COMBTYPE'] = ( 'MEAN'       , 'How raw frames were combined' )
    
    # List of header keys to copy over from the first raw dark file
    keys_to_copy = [ 'DATE', 'TIMEDAY', 'PLUS',                         # when first dark frame was taken
                     'SNAP_VER', 'SNAPDATE', 'DEVICE', 'PARTNUM',       # versioning, if ever wanted
                     'DCFILE', 'INITFILE',                              # ref files of potential interest
                     'WINTRANS', 'DETPITCH', 'APERDIST', 'APERDIAM',    # some info about exposures, if wanted
                     'FRMRATE', 'INTEGRT', 'INTEGRTM',                  # frame rate and integration
                     'GAIN_SET', 'CH0POWER', 'CH1POWER', 'CH2POWER', 'CH3POWER', 'CH4POWER', 'CH5POWER' ]
    
    # Opens the first raw dark file used to create the mean and copies some values from its header to the
    #   new header. Assumes these are in the 0th extension, not the data ext
    with fits.open( os.path.join( raw_cals_path, filelist[0] ) ) as raw_ref_hdu:
        for key in keys_to_copy:
            if key in raw_ref_hdu[0].header.keys():
                hdu.header[key] = raw_ref_hdu[0].header.cards[key][1:]
    
    # Finally, write this hdu to the output file
    hdu.writeto( os.path.join( calib_outpath, outfile_name ) )
    
    
    
    