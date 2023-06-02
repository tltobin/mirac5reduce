################## Importing packages ####################

from .. import __version__

from astropy.io import fits
import numpy as np
import configparser
import os

from ..utils.statfunc import medabsdev

################## Functions ####################

def make_bpmask( config, logfile = None, debug = False, **kwargs ):
    """
    Creates a bad pixel mask from an existing mean dark frame file, where pixels with mean dark value that
    deviates from the median value by more than some threshold times the M.A.D.
    
    Required Parameters
    -------------------
    
            config          String
            
                                The file name(s) (with paths) of the configuration file.
    
    Optional Parameters: Config File Override
    -----------------------------------------
                                
            bp_threshold    Integer or float
                                
                                [ Default = Config file value for bp_threshold ]
                                
                                Threshold defining the values at which pixels will be marked as bad.
                                
                                Pixels with mean dark values more than [bp_threshold] x the M.A.D. from the
                                median pixel value in the mean dark frame will be masked in the bad pixel 
                                mask.
                                
            startno         Integer
                                
                                [ Default = Config file value for raw_dark_startno ]
                                
                                File number of the first raw dark frame (inclusive). Used only to determine
                                default values for outfile and darkfile, below.
            
            endno           Integer
                                
                                [ Default = Config file value for raw_dark_startno ]
                                
                                File number of the last raw dark frame (inclusive). Used only to determine
                                default values for outfile and darkfile, below.
            
            outpath         String
                                
                                [ Default = Config file value for calib_outpath ]
                                
                                Path where the output files created by the calibration scripts are saved.
                                Mean dark file should be located in this path. Resulting bpmask file will also 
                                be saved to this path.
            
            outfile         String
                                
                                [ Default = bpmask_[startno]_[endno].fits ]
                                
                                File name used for the created bad pixel mask fits file. File is saved within 
                                the outpath.
            
            dark_file       String
                                
                                [ Default = dark_[startno]_[endno].fits ]
                                
                                File name for the mean dark file that will be used to determine the bad pixel 
                                locations. File should be located in the outpath.
    
    
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
        
        Sometimes used (see Optional Parameters above for more info):
            
            [CALIB]         calib_outpath
                            raw_dark_startno
                            raw_dark_endno
                            bp_threshold
                            
    Output Files Generated
    ----------------------
    
        [outpath]/[outfile]
        
                            Fits file containing (in extension 0) the generated bad pixel mask. 
                            
                            Copies some info from the file header of the mean dark file that it is derived
                            from, as well as the statistical values used to create it.
    """
    
    
    
    # Retrieves config file
    conf = configparser.ConfigParser()
    _ = conf.read(config)
    
    # Parses optional keys that may have been provided in **kwargs to override the config values and sets any 
    #   default values
    bp_threshold = conf[  'CALIB'  ]['bp_threshold']
    if 'bp_threshold' in kwargs.keys():
        bp_threshold = kwargs['bp_threshold']
    else:
        bp_threshold = float( bp_threshold )
    
    if 'outfile' not in kwargs.keys() or 'dark_file' not in kwargs.keys():
        startno      = conf[  'CALIB'  ]['raw_dark_startno']
        if 'startno' in kwargs.keys():
            startno = kwargs['startno']
        elif 'dark_startno' in kwargs.keys():
            startno = kwargs['dark_startno']
        elif 'raw_dark_startno' in kwargs.keys():
            startno = kwargs['raw_dark_startno']
        else:
            startno = int( startno )
    
        endno        = conf[  'CALIB'  ]['raw_dark_endno']
        if 'endno' in kwargs.keys():
            endno = kwargs['endno']
        elif 'dark_endno' in kwargs.keys():
            endno = kwargs['dark_endno']
        elif 'raw_dark_endno' in kwargs.keys():
            endno = kwargs['raw_dark_endno']
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
        outfile = 'bpmask_{0}_{1}.fits'.format( startno, endno )
    
    dark_file = None
    if 'dark_file' in kwargs.keys():
        dark_file = kwargs['dark_file']
    else:
        dark_file = 'dark_{0}_{1}.fits'.format( startno, endno )
    
    
    
    
    # Debugging message checkpoint
    if debug:
        feedbacklines = ['MAKE_BPMASK.DEBUG        Parameters set manually or determined from config file:',
                         'MAKE_BPMASK.DEBUG            {0: >16} : {1}'.format( 'bp_threshold', bp_threshold ),
                         'MAKE_BPMASK.DEBUG            {0: >16} : {1}'.format( 'outpath', outpath ),
                         'MAKE_BPMASK.DEBUG            {0: >16} : {1}'.format( 'dark_file', dark_file ),
                         'MAKE_BPMASK.DEBUG            {0: >16} : {1}'.format( 'outfile', outfile ) ]
        for flin in feedbacklines:
            print(flin)
    
    # Writes quick note to logfile or terminal
    feedback_msg = 'MAKE_BPMASK:         Finding bad pixels...'
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    
    if debug:
        print('MAKE_BPMASK.DEBUG        Retrieving data from dark_file...')
    
    # Retrieves mean dark data array from the file
    dark_data_mean = fits.getdata( os.path.join( outpath, dark_file ), 0 )
    
    
    if debug:
        print('MAKE_BPMASK.DEBUG          - Shape of retrieved array : {0}'.format( dark_data_mean.shape ))
        print('MAKE_BPMASK.DEBUG        Calculating stats and threshold of retrieved dark array...')
        
    # Calculates median and m.a.d. pixel value of frame
    med_of_mean_dark = np.median( dark_data_mean )
    mad_of_mean_dark = medabsdev( dark_data_mean )
    
    # Calculate thresholds for this test -- upper and lower
    threshold_hidark = med_of_mean_dark + bp_threshold * mad_of_mean_dark
    threshold_lodark = med_of_mean_dark - bp_threshold * mad_of_mean_dark

    if debug:
        print('MAKE_BPMASK.DEBUG          - Upper threshold value : {0}'.format( threshold_hidark ))
        print('MAKE_BPMASK.DEBUG          - Lower threshold value : {0}'.format( threshold_lodark ))
        print('MAKE_BPMASK.DEBUG        Finding pixels outside these values...')
    
    # Create bad pix mask for all pixels with values above and below these thresholds
    bpmask_hidark = ( dark_data_mean > threshold_hidark )
    bpmask_lodark = ( dark_data_mean < threshold_lodark )
    
    # Combined bad pix mask
    bpmask = ( bpmask_hidark | bpmask_lodark )
    
    # Writes quick note to logfile or terminal
    feedback_msg = 'MAKE_BPMASK:         Bad pixel mask generated. Pixels Masked: {0}'.format( bpmask.sum() )
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    if debug:
        print('MAKE_BPMASK.DEBUG          - Pix exceeding upper threshold : {0}'.format( bpmask_hidark.sum() ))
        print('MAKE_BPMASK.DEBUG          - Pix below lower threshold     : {0}'.format( bpmask_lodark.sum() ))
    
    feedback_msg = 'MAKE_BPMASK:         Saving bad pixel mask to outfile.'
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '{0}\n'.format(feedback_msg) )
    else:
        print(feedback_msg)
    
    # Creates hdu to save to file with mean frame
    hdu = fits.PrimaryHDU( bpmask.astype(int) )
    
    # Populates header directly with general info 
    hdu.header['FILETYPE'] =   'Pixel Mask'
    hdu.header['DARKFILE'] = ( dark_file               , 'Dark file used'                            )
    hdu.header['NFLAGGED'] = ( bpmask.sum()            , 'Total number bad pixels flagged'           )
    
    hdu.header['NSIG'    ] = ( bp_threshold            , 'Threshold x M.A.D. used (bp_threshold)'    )
    hdu.header['DARKMAD' ] = ( mad_of_mean_dark        , 'Med. Abs. Dev. (M.A.D.) of mean dark (DN)' )
    hdu.header['DARKMED' ] = ( med_of_mean_dark        , 'Median pix value of mean dark (DN)'        )
    hdu.header['CUTOVER' ] = ( threshold_hidark        , 'Upper limit for good pix values (DN)'      )
    hdu.header['NFLGOVER'] = ( bpmask_hidark.sum()     , 'Number pixels flagged for values > CUTOVER')
    hdu.header['CUTUNDR' ] = ( threshold_lodark        , 'Lower limit for good pix values (DN)'      )
    hdu.header['NFLGUNDR'] = ( bpmask_lodark.sum()     , 'Number pixels flagged for values < CUTUNDR')
    
    
    # Copies some header values from the mean dark file used
    with fits.open( os.path.join( outpath, dark_file ) ) as dark_hdu:
        hdu.header['NFRAMES' ] = ( dark_hdu[0].header['NFRAMES' ], 'Number raw frames in darkfile' )
        hdu.header['FILE_STR'] = ( dark_hdu[0].header['FILE_STR'], 'First raw file in darkfile' )
        hdu.header['FILE_END'] = ( dark_hdu[0].header['FILE_END'], 'Last raw file in darkfile' )
        hdu.header['COMBTYPE'] = ( dark_hdu[0].header['COMBTYPE'], 'How darkfile frames were combined' )
        
        # List of header keys to copy over directly that came from the first raw dark file
        keys_to_copy = [ 'DATE', 'TIMEDAY', 'PLUS',                         # when first dark frame was taken
                         'SNAP_VER', 'SNAPDATE', 'DEVICE', 'PARTNUM',       # versioning, if ever wanted
                         'DCFILE', 'INITFILE',                              # ref files of potential interest
                         'WINTRANS', 'DETPITCH', 'APERDIST', 'APERDIAM',    # some info about exposures, if wanted
                         'FRMRATE', 'INTEGRT', 'INTEGRTM',                  # frame rate and integration
                         'GAIN_SET', 'CH0POWER', 'CH1POWER', 'CH2POWER', 'CH3POWER', 'CH4POWER', 'CH5POWER' ]

        for key in keys_to_copy:
            if key in dark_hdu[0].header.keys():
                hdu.header[key] = dark_hdu[0].header.cards[key][1:]
    
    # Finally, write this hdu to the output file
    hdu.writeto( os.path.join( outpath, outfile ) )
    
    
    
        
