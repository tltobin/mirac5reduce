################## Importing packages ####################

from .. import __version__

import os
import numpy as np
import configparser

from ..utils.utils import get_raw_filenames, write_mean_frame, write_chopnod_frame
from ..utils.calcframes import calc_mean_frame, calc_chopnod_frame

################## Functions ####################

def meanframe( config, frametype, logfile = None, debug = False, **kwargs ):
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
    
    Optional Parameters: Config File Override
    -----------------------------------------
            
            datapath        String
                                
                                [ Default = Config file value for raw_(dark/flat/data)_path ]
                                
                                The path where the raw fits frame fits files are stored.
                                
                                If frametype is 'dark', uses config value, raw_dark_path. 
                                
                                If frametype is 'flat', uses config value, raw_flat_path.
                                
                                Otherwise, uses config value, raw_data_path.
                                
            startno         Integer
                                
                                [ Default = Config file value for 
                                  raw_dark_startno/raw_flat_startno/raw_data_startno ]
                                
                                File number of the first file in the range of files to be averaged together.
                                
                                If frametype is 'dark', uses config value, raw_dark_startno.
                                
                                If frametype is 'flat', uses config value, raw_flat_startno.
                                
                                Otherwise, uses config value, raw_data_startno.
                                
            endno           Integer
                                
                                [ Default = Config file value for 
                                  raw_dark_endno/raw_flat_endno/raw_data_endno ]
                                
                                File number of the last file in the range of files to be averaged together.
                                
                                If frametype is 'dark', uses config value, raw_dark_endno.
                                
                                If frametype is 'flat', uses config value, raw_flat_endno.
                                
                                Otherwise, uses config value, raw_data_endno.
            
            outpath         String
                                
                                [ Default = Config file value for calib_outpath/reduce_outpath ]
                                
                                The path to which the output file created by this function will be saved.
                                
                                If frametype is 'dark' or 'flat', uses config value, calib_outpath.
                                
                                Otherwise, uses config value, reduce_outpath.
            
            outfile         String
                                
                                [ Default = [frametype]_[startno]_[endno].fits ]
                                
                                File name used for the created mean frame fits file. File is saved within 
                                the outpath.
                                
    Optional Parameters: Other
    --------------------------
            
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
            
            [CALIB]         raw_dark_path
                            raw_dark_startno
                            raw_dark_endno
                            raw_flat_path
                            raw_flat_startno
                            raw_flat_endno
                            calib_outpath
                            
    Output Files Generated
    ----------------------
    
        [outpath]/[outfile]
        
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
    
    # Parses optional keys that may have been provided in **kwargs to override the config values and sets any 
    #   default values based on the provided frametype
    frametype = frametype.lower()
    
    if frametype in ['dark','flat']:
        
        datapath     = conf[  'CALIB'  ]['raw_{0}_path'.format(frametype)]
        if 'datapath' in kwargs.keys():
            datapath = kwargs['datapath']
        elif '{0}_path'.format(frametype) in kwargs.keys():
            datapath = kwargs['{0}_path'.format(frametype)]
        elif 'raw_{0}_path'.format(frametype) in kwargs.keys():
            datapath = kwargs['raw_{0}_path'.format(frametype)]
    
        startno      = conf[  'CALIB'  ]['raw_{0}_startno'.format(frametype)]
        if 'startno' in kwargs.keys():
            startno = kwargs['startno']
        elif '{0}_startno'.format(frametype) in kwargs.keys():
            startno = kwargs['{0}_startno'.format(frametype)]
        elif 'raw_{0}_startno'.format(frametype) in kwargs.keys():
            startno = kwargs['raw_{0}_startno'.format(frametype)]
        else:
            startno = int( startno )
    
        endno        = conf[  'CALIB'  ]['raw_{0}_endno'.format(frametype)]
        if 'endno' in kwargs.keys():
            endno = kwargs['endno']
        elif '{0}_endno'.format(frametype) in kwargs.keys():
            endno = kwargs['{0}_endno'.format(frametype)]
        elif 'raw_{0}_endno'.format(frametype) in kwargs.keys():
            endno = kwargs['raw_{0}_endno'.format(frametype)]
        else:
            endno = int( endno )
    
        outpath      = conf[  'CALIB'  ]['calib_outpath']
        if 'outpath' in kwargs.keys():
            outpath = kwargs['outpath']
        elif 'calib_outpath' in kwargs.keys():
            outpath = kwargs['calib_outpath']
    
        outfile = None
        if 'outfile' in kwargs.keys():
            outfile = kwargs['outfile']
        else:
            outfile = '{0}_{1}_{2}.fits'.format( frametype, startno, endno )
    
    else:
        
        datapath     = conf['REDUCTION']['raw_data_path']
        if 'datapath' in kwargs.keys():
            datapath = kwargs['datapath']
        elif 'raw_data_path' in kwargs.keys():
            datapath = kwargs['raw_data_path']
    
        startno      = conf['REDUCTION']['raw_data_startno']
        if 'startno' in kwargs.keys():
            startno = kwargs['startno']
        elif 'data_startno' in kwargs.keys():
            startno = kwargs['data_startno']
        elif 'raw_data_startno' in kwargs.keys():
            startno = kwargs['raw_data_startno']
        else:
            startno = int( startno )
    
        endno        = conf['REDUCTION']['raw_data_endno']
        if 'endno' in kwargs.keys():
            endno = kwargs['endno']
        elif 'data_endno' in kwargs.keys():
            endno = kwargs['data_endno']
        elif 'raw_data_endno' in kwargs.keys():
            endno = kwargs['raw_data_endno']
        else:
            endno = int( endno )
    
        outpath      = conf['REDUCTION']['reduce_outpath']
        if 'outpath' in kwargs.keys():
            outpath = kwargs['outpath']
        elif 'reduce_outpath' in kwargs.keys():
            outpath = kwargs['reduce_outpath']
    
        outfile = None
        if 'outfile' in kwargs.keys():
            outfile = kwargs['outfile']
        else:
            outfile = '{0}_{1}_{2}.fits'.format( frametype, startno, endno )
        
        
    
    
    # Debugging message checkpoint
    if debug:
        feedbacklines = ['MEANFRAME.DEBUG          Provided frametype: {0}'.format(frametype),
                         'MEANFRAME.DEBUG          Parameters set manually or determined from frametype:',
                         'MEANFRAME.DEBUG              {0: >16} : {1}'.format( 'datapath', datapath ),
                         'MEANFRAME.DEBUG              {0: >16} : {1}'.format( 'startno', startno ),
                         'MEANFRAME.DEBUG              {0: >16} : {1}'.format( 'endno', endno ),
                         'MEANFRAME.DEBUG              {0: >16} : {1}'.format( 'outpath', outpath ),
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
    write_mean_frame( os.path.join( outpath, outfile ), avgframe, frametype, filelist, raw_filepath = datapath )
    


def chopnodframe( config, logfile = None, debug = False, **kwargs ):
    """
    Combines raw frame files (with file numbers ranging from startno and endno) into a single mean chop/nod 
    difference frame and saves the result to a fits file. 
    
    Note: This does assume that all raw frames are stored in their own fits files, under the extension
    indicated by data_ext in the config file.
    
    Required Parameters
    -------------------
    
            config          String
            
                                The file name(s) (with paths) of the configuration file.
    
    Optional Parameters: Config File Override
    -----------------------------------------
            
            datapath        String
                                
                                [ Default = Config file value for raw_data_path ]
                                
                                The path where the raw fits frame fits files are stored.
                                
            startno         Integer
                                
                                [ Default = Config file value for raw_data_startno ]
                                
                                File number of the first file in the range of chop/nod files to be combined.
                                
            endno           Integer
                                
                                [ Default = Config file value for raw_data_endno ]
                                
                                File number of the last file in the range of chop/nod files to be combined.
                                
            chopfreq        Float
                                
                                [ Default = Config file value for chopfreq ]
                                
                                Frequency (in Hz) at which chopping occurs. This is converted into number of 
                                frame files using header keys in the raw data files.
                                
            nodfreq         Float or None
                                
                                [ Default = Config file value for nodfreq ]
                                
                                Frequency (in Hz) at which nodding occurs. This is converted into number of 
                                frame files using header keys in the raw data files.
            
            outpath         String
                                
                                [ Default = Config file value for reduce_outpath ]
                                
                                The path to which the output file created by this function will be saved.
            
            outfile         String
                                
                                [ Default = chopnod_[startno]_[endno].fits ]
                                
                                File name used for the created mean chop-nod subtracted frame fits file. File 
                                is saved within the outpath.
                            
     Optional Parameters: Other
    ---------------------------
            
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
                            raw_data_startno
                            raw_data_endno
                            chopfreq
                            nodfreq
                            reduce_outpath
                            
    Output Files Generated
    ----------------------
    
        [outpath]/[outfile]
        
                            Fits file containing (in extension 0) the calculated mean chop/nod difference 
                            frame. 
                            
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
    
    # Parses optional keys that may have been provided in **kwargs to override the config values and sets any 
    #   default values
    datapath     = conf['REDUCTION']['raw_data_path']
    if 'datapath' in kwargs.keys():
        datapath = kwargs['datapath']
    elif 'raw_data_path' in kwargs.keys():
        datapath = kwargs['raw_data_path']
    
    startno      = conf['REDUCTION']['raw_data_startno']
    if 'startno' in kwargs.keys():
        startno = kwargs['startno']
    elif 'data_startno' in kwargs.keys():
        startno = kwargs['data_startno']
    elif 'raw_data_startno' in kwargs.keys():
        startno = kwargs['raw_data_startno']
    else:
        startno = int( startno )
    
    endno        = conf['REDUCTION']['raw_data_endno']
    if 'endno' in kwargs.keys():
        endno = kwargs['endno']
    elif 'data_endno' in kwargs.keys():
        endno = kwargs['data_endno']
    elif 'raw_data_endno' in kwargs.keys():
        endno = kwargs['raw_data_endno']
    else:
        endno = int( endno )
    
    chopfreq     = conf['REDUCTION']['chopfreq']
    if 'chopfreq' in kwargs.keys():
        chopfreq = kwargs['chopfreq']
    chopfreq = float( chopfreq )
    
    nodfreq      = conf['REDUCTION']['nodfreq']
    if 'nodfreq' in kwargs.keys():
        nodfreq = kwargs['nodfreq']
    else:
        try:
            nodfreq = float( nodfreq )
        except:
            nodfreq = None
    
    outpath      = conf['REDUCTION']['reduce_outpath']
    if 'outpath' in kwargs.keys():
        outpath = kwargs['outpath']
    elif 'reduce_outpath' in kwargs.keys():
        outpath = kwargs['reduce_outpath']
    
    outfile = None
    if 'outfile' in kwargs.keys():
        outfile = kwargs['outfile']
    else:
        outfile = 'chopnod_{0}_{1}.fits'.format( startno, endno )
    
    
    
    # Debugging message checkpoint
    if debug:
        feedbacklines = ['CHOPNODFRAME.DEBUG       Parameters set manually or determined from config file:',
                         'CHOPNODFRAME.DEBUG           {0: >16} : {1}'.format( 'datapath', datapath ),
                         'CHOPNODFRAME.DEBUG           {0: >16} : {1}'.format( 'startno', startno ),
                         'CHOPNODFRAME.DEBUG           {0: >16} : {1}'.format( 'endno', endno ),
                         'CHOPNODFRAME.DEBUG           {0: >16} : {1}'.format( 'chopfreq', chopfreq ),
                         'CHOPNODFRAME.DEBUG           {0: >16} : {1}'.format( 'nodfreq', nodfreq ),
                         'CHOPNODFRAME.DEBUG           {0: >16} : {1}'.format( 'outpath', outpath ),
                         'CHOPNODFRAME.DEBUG           {0: >16} : {1}'.format( 'outfile', outfile ),
                         'CHOPNODFRAME.DEBUG       Parameters retrieved from config file:',
                         'CHOPNODFRAME.DEBUG           {0: >16} : {1}'.format( 'save_mem', save_mem ),
                         'CHOPNODFRAME.DEBUG           {0: >16} : {1} -> {2}'.format( 
                                'max_frames_inmem', conf['COMPUTING']['max_frames_inmem'], max_frames_inmem ),
                         'CHOPNODFRAME.DEBUG           {0: >16} : {1} -> {2}'.format( 
                                'data_ext', conf['DATA_ARCH']['data_ext'], data_ext ),
                         'CHOPNODFRAME.DEBUG           {0: >16} : {1}'.format( 'raw_name_fmt', raw_name_fmt ) ]
        for flin in feedbacklines:
            print(flin)
    
    # Writes quick note to logfile or terminal
    feedback_msg = 'CHOPNODFRAME:        Retrieving list of files with numbers {0}-{1}.'.format( startno, endno )
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    # Retrieves the list of file names for the requested file numbers
    filelist = get_raw_filenames( raw_name_fmt, startno, endno, datapath  )
    
    
    
    # Debugging message checkpoint
    if debug:
        print('CHOPNODFRAME.DEBUG       File names retrieved: {0}'.format(len(filelist)))
    
    # Writes quick note to logfile or terminal regarding whether memory saving is turned on or not for this
    feedback_msg = 'CHOPNODFRAME:        Calculating chop/nod mean diff frame with save_mem = {0}'.format( str(save_mem) )
    if save_mem: 
        feedback_msg += ' (max {0} frames)'.format( max_frames_inmem )
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    
    # Splits here to calculate average frame from data files in memory saving mode or directly
    if save_mem:
        diffframe, header_dict = calc_chopnod_frame( [ os.path.join( datapath, fname ) for fname in filelist ],
                                                      chopfreq = chopfreq, nodfreq = nodfreq, 
                                                      ext = data_ext, maxframes = max_frames_inmem, logfile = logfile,
                                                      _fitsdict_ = True )
    else:
        diffframe, header_dict = calc_chopnod_frame( [ os.path.join( datapath, fname ) for fname in filelist ], 
                                                      chopfreq = chopfreq, nodfreq = nodfreq, 
                                                      ext = data_ext, maxframes = None, logfile = logfile,
                                                      _fitsdict_ = True )
    

    
    # Writes quick note to logfile or terminal
    feedback_msg = 'CHOPNODFRAME:        Saving Results to {0}.'.format( outfile )
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    
    # Uses write_chopnod_frame function to save the calculated data to the desired fits file
    write_chopnod_frame( os.path.join( outpath, outfile ), diffframe, filelist, raw_filepath = datapath, header_dict = header_dict )
    
    
    