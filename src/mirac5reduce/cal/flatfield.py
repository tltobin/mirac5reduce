################## Importing packages ####################

from .. import __version__

from astropy.io import fits
import numpy as np
import configparser
from math import ceil
import os
from datetime import datetime

from ..utils.statfunc import medabsdev
from ..utils.utils import get_raw_filenames

################## Functions ####################

def create_flatfield( config, 
                      datapath = None, startno = None, endno = None, outfile = None, 
                      dark_file = None, bpmask_file = None, 
                      logfile = None, debug = False ):
    """
    Creates a scaled flatfield frame from raw flat frames and a mean dark frame, and saves to an output fits 
    file.
    
    Raw flat frames are dark subtracted, scaled by that frame's median (calculated with the bad pixel mask 
    applied, if any), and averaged together.
    
    Assumes all raw files are stored in separate fits files, with one file per raw 2D frame.
    
    Note on memory saving: This function is expected to require about twice as much memory per frame as other
    simpler functions like calcframes.calc_mean_frame when a bad pixel mask is applied. If memory saving 
    measures are requested by the config file (save_mem = True) and a bad pixel mask file is used, the maximum 
    number of raw frames processed simultaneously will be *half* that specified by config value, 
    max_frames_inmem, to compensate for this.
    
    Required Parameters
    -------------------
    
            config          String
            
                                The file name(s) (with paths) of the configuration file.
            
    Optional Parameters
    -------------------
            
            datapath        String or None
                                
                                [ Default = None ]
                                
                                The path where the raw flat frame fits files are stored.
                                
                                If set to None, will use value specified in the config file as raw_cals_path.
                                
            startno         Integer or None
                                
                                [ Default = None ]
                                
                                File number of the first raw flat frame.
                                
                                If set to None, will use value specified in the config file as 
                                raw_flat_startno.
            
            endno           Integer or None
                                
                                [ Default = None ]
                                
                                File number of the last raw flat frame.
                                
                                If set to None, will use value specified in the config file as raw_flat_endno.
            
            outfile         String or None
                                
                                [ Default = None ]
                                
                                Path and name to which the fits file with the resulting flatfield frame will 
                                be saved. 
                                
                                If set to None, will derive using values imported from the provided config
                                file:
                                outfile = '[calib_outpath]/flatfield_[startno]_[endno].fits'.
            
            dark_file       String or None
                                
                                [ Default = None ]
                                
                                The path and file name for the mean dark file that is closest in time to the
                                raw flat frames specified.
                                
                                If set to None, will derive using values imported from the provided config 
                                file:
                                
                                [calib_outpath]/dark_[raw_dark_startno]_[raw_dark_endno].fits
            
            bpmask_file     String or None
                                
                                [ Default = None ]
                                
                                The path and file name for the bad pixel file calculated from the provided 
                                dark_file. Used to mask outstanding pixel values when calculating the median
                                of each frame, which is used to normalize them before combining. Final saved
                                flatfield frame is not masked.
                                
                                If set to None, will derive using values imported from the provided config 
                                file:
                                [calib_outpath]/bpmask_[raw_dark_startno]_[raw_dark_endno].fits
                                
                                If that file does not exist, no masking will occur.
            
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
            
            [CALIB]         calib_outpath
                            raw_dark_startno
                            raw_dark_endno
                            raw_flat_startno
                            raw_flat_endno
                            
    Output Files Generated
    ----------------------
    
        [outfile]
        
                            Fits file containing (in extension 0) the generated flatfield frame.
                            
                            Copies some info from the file header of the raw flat files and the mean dark file 
                            that it was derived from.
    
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
    #   values
    if datapath is None:
        datapath = conf[  'CALIB'  ]['raw_cals_path']
    if startno is None:
        startno  = conf[  'CALIB'  ].getint('raw_flat_startno')
    if endno is None:
        endno    = conf[  'CALIB'  ].getint('raw_flat_endno')
    if outfile is None:
        outpath  = conf[  'CALIB'  ]['calib_outpath']
        outfile  = os.path.join( outpath, 'flatfield_{0}_{1}.fits'.format(startno, endno) )
    if dark_file is None:
        outpath  = conf[  'CALIB'  ]['calib_outpath']
        dark_startno = conf[  'CALIB'  ].getint('raw_dark_startno')
        dark_endno   = conf[  'CALIB'  ].getint('raw_dark_endno')
        dark_file    = os.path.join( outpath, 'dark_{0}_{1}.fits'.format(dark_startno, dark_endno) )
    if bpmask_file is None:
        outpath  = conf[  'CALIB'  ]['calib_outpath']
        dark_startno = conf[  'CALIB'  ].getint('raw_dark_startno')
        dark_endno   = conf[  'CALIB'  ].getint('raw_dark_endno')
        bpmask_file    = os.path.join( outpath, 'bpmask_{0}_{1}.fits'.format(dark_startno, dark_endno) )
        if not os.path.isfile( bpmask_file ):
            bpmask_file = None
    
        
    
    
    # Debugging message checkpoint
    if debug:
        feedbacklines = ['CREATE_FLATFIELD.DEBUG   Parameters set manually or determined from config file:',
                         'CREATE_FLATFIELD.DEBUG       {0: >16} : {1}'.format( 'datapath', datapath ),
                         'CREATE_FLATFIELD.DEBUG       {0: >16} : {1}'.format( 'startno', startno ),
                         'CREATE_FLATFIELD.DEBUG       {0: >16} : {1}'.format( 'endno', endno ),
                         'CREATE_FLATFIELD.DEBUG       {0: >16} : {1}'.format( 'outfile', outfile ),
                         'CREATE_FLATFIELD.DEBUG       {0: >16} : {1}'.format( 'dark_file', dark_file ),
                         'CREATE_FLATFIELD.DEBUG       {0: >16} : {1}'.format( 'bpmask_file', bpmask_file ),
                         'CREATE_FLATFIELD.DEBUG   Parameters retrieved from config file:',
                         'CREATE_FLATFIELD.DEBUG       {0: >16} : {1}'.format( 'save_mem', save_mem ),
                         'CREATE_FLATFIELD.DEBUG       {0: >16} : {1} -> {2}'.format( 
                                'max_frames_inmem', conf['COMPUTING']['max_frames_inmem'], max_frames_inmem ),
                         'CREATE_FLATFIELD.DEBUG       {0: >16} : {1} -> {2}'.format( 
                                'data_ext', conf['DATA_ARCH']['data_ext'], data_ext ),
                         'CREATE_FLATFIELD.DEBUG       {0: >16} : {1}'.format( 'raw_name_fmt', raw_name_fmt ) ]
        for flin in feedbacklines:
            print(flin)
    
    # Writes quick note to logfile or terminal
    feedback_msg = 'CREATE_FLATFIELD:    Retrieving list of files with numbers {0}-{1}.'.format( startno, endno )
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    
    
    
    # Retrieves the list of file names for the requested file numbers
    filelist = get_raw_filenames( raw_name_fmt, startno, endno, datapath  )
    
    # feedback message checkpoint
    feedback_msg = 'CREATE_FLATFIELD:        File names retrieved: {0}'.format(len(filelist))
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    
    
    
    # Loads in mean dark array
    feedback_msg = 'CREATE_FLATFIELD:    Loading master dark from dark_file.'
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    # Retrieves array from dark file ext 0
    dark_array = fits.getdata( dark_file, 0 )
    
    
    
    # If mask file was provided, reads it in
    if bpmask_file is not None:
        
        # Prints feedback
        feedback_msg = 'CREATE_FLATFIELD:    Loading bad pixel mask from bpmask_file.'
        if logfile is not None:
            with open(logfile,'a') as lf:
                lf.write( '{0}\n'.format(feedback_msg) )
        else:
            print(feedback_msg)
        
        # Retrieves array from bpmask file ext 0
        bpmask_array = fits.getdata( bpmask_file, 0 )
    
    # Otherwise, prints a message and does basic bookkeeping
    else:
        feedback_msg = 'CREATE_FLATFIELD:    No bad pixel mask provided.'
        if logfile is not None:
            with open(logfile,'a') as lf:
                lf.write( '{0}\n'.format(feedback_msg) )
        else:
            print(feedback_msg)
        bpmask_array = None
    
    
        
        
    
    
    
    # Writes quick note to logfile or terminal regarding whether memory saving is turned on or not for this
    feedback_msg = 'CREATE_FLATFIELD:    Preparing calculation with save_mem = {0}'.format( str(save_mem) )
    if save_mem: 
        feedback_msg += ' (max {0} frames)'.format( max_frames_inmem )
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    
    
    
    
    
    
    
    ## Procedes explicitly here instead of calling separate function to do this calculation from utils ##
    ##           (means we're assuming all raw frames are in their own, individual fits files)         ##
    
    # Retrieves the shape of a single 2D frame
    with fits.open( os.path.join( datapath, filelist[0] ), mode='readonly' ) as hdulist:
        frame_shape = hdulist[data_ext].data.shape
    
    # Sets aside total number of frames
    totframes = len( filelist )
    
    
    
    # If max number of frames was provided (as not None), determines how it needs to be split up into chunks
    if max_frames_inmem is not None:
        
        # maxframes processed are only half those allowed in memory if bad pix mask applied
        if bpmask_file is not None:
            maxframes = int( max_frames_inmem / 2 )
        else:
            maxframes = max_frames_inmem
        
        # Creates arrays of number of frames per chunk and the associated weights to use for each chunk of 
        #   frames when added to final combined array 
        nchunks = ceil( totframes / maxframes )
        nframes = np.array( [ maxframes, ]*nchunks )
        if ( totframes % maxframes ) != 0:
            nframes[-1] = ( totframes % maxframes )
        chunkweights = nframes / totframes
        loopframes = maxframes
    
        # Before starting, prints some feedback to log or terminal
        feedbacklines = [        'CREATE_FLATFIELD:        {0} Frames of shape {1}'.format( totframes, frame_shape ),
                                 '                         Max {0} frames loaded simultaneously ({1} chunks)'.format(maxframes, nchunks) ]
        if np.sum( chunkweights ) != 1.0:
            tmpsum = np.sum( chunkweights )
            feedbacklines.append('                     Warning: Chunk weights do not sum to 1. Actual Sum: {0} (diff {1:.2e})'.format( tmpsum, 1.-tmpsum )  )
        if logfile is not None:
            with open(logfile,'a') as lf:
                for i in range(len(feedbacklines)):
                    lf.write( '{0}\n'.format( feedbacklines[i] ) )
        else:
            for i in range(len(feedbacklines)):
                print( feedbacklines[i] )
    
    # If there is no limit on number of frames that can be read in, just has single chunk with all frames
    else:
    
        # Creates same variables as memory-saving version 
        nchunks = 1
        nframes = np.array([ totframes, ])
        chunkweights = nframes / totframes
        loopframes = 0
        
        # Before starting, prints some feedback to log or terminal
        feedback_msg =          'CREATE_FLATFIELD:        {0} Frames of shape {1}'.format( totframes, frame_shape )
        if logfile is not None:
            with open(logfile,'a') as lf:
                lf.write( '{0}\n'.format(feedback_msg) )
        else:
            print(feedback_msg)
    
    
    
    # Create the flatfield frame to populate 
    flatfield = np.zeros( frame_shape )
    
        
    # Prints feedback
    feedback_msg = 'CREATE_FLATFIELD:    Creating flatfield image'
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( feedback_msg )
    else:
        print(feedback_msg, end='')

    
    
    # Actually iterates through frames, reading them in by chunks, subtracting the darks, calculating the
    #   (masked median) and normalizing, and adding their weighted mean to the flatfield array
    for i, nframes_in_chunk in enumerate( nframes ):
        
        # Adds to feedback one period per chunk to track progress
        if logfile is not None:
            with open(logfile,'a') as lf:
                lf.write('.')
        else:
            print('.', end='')
        
        # Determines the indices of the files in filelist that will be read in for that chunk of frames
        file_idx_str = i * loopframes
        file_idx_end = file_idx_str + nframes_in_chunk
        
        # Retrieves the data from those files and builds them into a list, which will be turned into an 
        #   array with frame as the 0th axis
        chunk_frames = list()
        for j in range( file_idx_str, file_idx_end ):
            chunk_frames.append( fits.getdata( os.path.join( datapath, filelist[j] ), data_ext, header = False ) )
        chunk_frames = np.array( chunk_frames )
        
        # Subtracts master dark from each frame
        chunk_frames = chunk_frames - dark_array[np.newaxis,:,:]
        
        
        # Does median calculation separately depending on whether we're masking bad pix or not
        if bpmask_file is not None:
            chunk_frames = np.ma.masked_array( chunk_frames, mask = chunk_frames*bpmask_array[np.newaxis,:,:] )
            med_vals = np.nanmedian( chunk_frames.filled( np.nan ), axis = (1,2) )
            chunk_frames = chunk_frames.data
        else:
            med_vals = np.nanmedian( chunk_frames, axis = (1,2) )
        
        
        
        # Divides each frame by its (masked) median
        chunk_frames = chunk_frames / med_vals[:,np.newaxis,np.newaxis]
        
        
        
        # Then calculate the mean frame, weight it, and add it to the flatfield
        mean_chunk = np.nanmean( chunk_frames, axis=0 )
        flatfield += chunkweights[i] * mean_chunk
    
    
    
    # Tidies up feedback lines
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write('Done.\n')
    else:
        print('Done.')
    
    
    
    ## Again, explicitly writes output fits file instead of calling lower level function from utils ##
    
    # Brief update in feedback log/terminal
    feedback_msg = 'CREATE_FLATFIELD:    Saving flatfield to outfile.'
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    # Creates hdu to save to file with mean frame
    hdu = fits.PrimaryHDU( flatfield )
    
    # Populates header directly with general info 
    hdu.header['FILETYPE'] =   'Flatfield'
    hdu.header['NFRAMES' ] = ( len(filelist)           , 'Number raw flat frames used' )
    hdu.header['FILE_STR'] = ( filelist[0]             , 'First raw flat file used'    )
    hdu.header['FILE_END'] = ( filelist[-1]            , 'Last raw flat file used'     )
    hdu.header['DARKFILE'] = ( dark_file.split('/')[-1], 'Dark file used' )
    
    
    # Copies some header values from the mean dark file used
    with fits.open( dark_file ) as dark_hdu:
        hdu.header['DKFRAMES'] = ( dark_hdu[0].header['NFRAMES' ], 'Number raw frames in darkfile' )
        hdu.header['DK_STR'  ] = ( dark_hdu[0].header['FILE_STR'], 'First raw file in darkfile' )
        hdu.header['DK_END'  ] = ( dark_hdu[0].header['FILE_END'], 'Last raw file in darkfile' )
        hdu.header['DK_COMB' ] = ( dark_hdu[0].header['COMBTYPE'], 'How darkfile frames were combined' )
    
    # Copies some header keys from the bad pixel file, if one was used
    if bpmask_file is not None:
        hdu.header['MASKFILE'] = ( bpmask_file.split('/')[-1], 'Bad pix mask file used' )
        with fits.open( bpmask_file ) as bpmask_hdu:
            hdu.header['MASKDARK'] = ( bpmask_hdu[0].header['DARKFILE' ], 'Dark file used to create bpmask'       )
            hdu.header['MASKNPIX'] = ( bpmask_hdu[0].header['NFLAGGED' ], 'Total number of pix flagged in bpmask' )
            hdu.header['MASKNSIG'] = ( bpmask_hdu[0].header['NSIG'     ], 'bp_threshold used to make bpmask'      )
    # If no bad pix file used, adds keys but empty
    else:
        hdu.header['MASKFILE'] = ( None, 'Bad pix mask file used' )
        hdu.header['MASKDARK'] = ( None, 'Dark file used to create bpmask'       )
        hdu.header['MASKNPIX'] = ( None, 'Total number of pix flagged in bpmask' )
        hdu.header['MASKNSIG'] = ( None, 'bp_threshold used to make bpmask'      )
    
    
    # Copies header keys from first raw fits file used to make flatfield
    keys_to_copy = [ 'DATE', 'TIMEDAY', 'PLUS',                         # when first dark frame was taken
                     'SNAP_VER', 'SNAPDATE', 'DEVICE', 'PARTNUM',       # versioning, if ever wanted
                     'DCFILE', 'INITFILE',                              # ref files of potential interest
                     'WINTRANS', 'DETPITCH', 'APERDIST', 'APERDIAM',    # some info about exposures, if wanted
                     'FRMRATE', 'INTEGRT', 'INTEGRTM',                  # frame rate and integration
                     'GAIN_SET', 'CH0POWER', 'CH1POWER', 'CH2POWER', 'CH3POWER', 'CH4POWER', 'CH5POWER' ]
    with fits.open( os.path.join(datapath,filelist[0]), mode='readonly' ) as rawhdu:
        for key in keys_to_copy:
            if key in rawhdu[0].header.keys():
                hdu.header[key] = rawhdu[0].header.cards[key][1:]
    
    # Finally, write some notes to the HISTORY cards of the header
    hdu.header['HISTORY']     = 'mirac5reduce version : {0}'.format( __version__ )
    hdu.header['HISTORY']     = 'Function : {0}'.format( 'cal.flatfield.create_flatfield' )
    hdu.header['HISTORY']     = '    Raw Data Arch : raw_name_fmt = {0}, data_ext = {1}'.format( raw_name_fmt, data_ext )
    hdu.header['HISTORY']     = '    Raw flat files : {0} files, file numbers {1} - {2}'.format( len(filelist), startno, endno )
    hdu.header['HISTORY']     = '    Subtracted mean dark from each flat frame : {0}'.format( dark_file )
    if bpmask_file is not None:
        hdu.header['HISTORY'] = '    Normalized each by masked median pix value in frame'
        hdu.header['HISTORY'] = '      - Bad pixel mask : {0}'.format( bpmask_file )
    else:
        hdu.header['HISTORY'] = '     Normalized each by median pix value in frame'
    hdu.header['HISTORY']     = '    Average dark subtracted and normalized flat frames'
    hdu.header['HISTORY']     = '    Flatfield file created : {0}'.format( datetime.now().strftime('%Y-%m-%d %H:%M:%S') )
    
    
    # Finally, write this hdu to the output file
    hdu.writeto( outfile )





