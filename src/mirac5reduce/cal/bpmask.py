################## Importing packages ####################

from astropy.io import fits
import numpy as np
import configparser

from ..utils.statfunc import medabsdev

################## Functions ####################

def make_bpmask( config, 
                 dark_file = None, bp_threshold = None, outfile = None,
                 logfile = None, debug = False  ):
    """
    Creates a bad pixel mask from an existing mean dark frame file, where pixels with mean dark value that
    deviates from the median value by more than some threshold times the M.A.D.
    
    Required Parameters
    -------------------
    
            config          String
            
                                The file name(s) (with paths) of the configuration file.
            
    Optional Parameters
    -------------------
            
            dark_file       String or None
                                
                                [ Default = None ]
                                
                                The path and file name for the mean dark file that will be used to determine
                                the bad pixel locations.
                                
                                If set to None, will derive using values imported from the provided config 
                                file:
                                
                                [calib_outpath]/dark_[raw_dark_startno]_[raw_dark_endno].fits
            
            bp_threshold    Float, Integer, or None
                                
                                [ Default = None ]
                                
                                Pixels with mean dark values more than [bp_threshold] x the M.A.D. from the
                                median pixel value in the mean dark frame will be masked in the bad pixel 
                                mask.      
                                                          
                                If set to None, will use the config file value, bp_threshold.
            
            outfile         String or None
                                
                                [ Default = None ]
                                
                                The path and file name where the fits file containing the created bad pixel 
                                mask will be saved.
                                
                                If set to None, will derive using values imported from the provided config 
                                file:
                                
                                [calib_outpath]/bpmask_[raw_dark_startno]_[raw_dark_endno].fits
            
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
    
        [outfile]
        
                            Fits file containing (in extension 0) the generated bad pixel mask. 
                            
                            Copies some info from the file header of the mean dark file that it is derived
                            from, as well as the statistical values used to create it.
    """
    
    
    
    # Retrieves config file
    conf = configparser.ConfigParser()
    _ = conf.read(config)
    
    # Parses optional keys that may have been provided to override the config values and sets any default
    #   values
    if dark_file is None:
        dark_file = '{0}/dark_{1}_{2}.fits'.format( conf['CALIB']['calib_outpath'], 
                                        conf['CALIB']['raw_dark_startno'], conf['CALIB']['raw_dark_endno'] )
    if bp_threshold is None:
        bp_threshold = conf['CALIB'].getfloat('bp_threshold')
    if outfile is None:
        outfile = '{0}/bpmask_{1}_{2}.fits'.format( conf['CALIB']['calib_outpath'], 
                                        conf['CALIB']['raw_dark_startno'], conf['CALIB']['raw_dark_endno'] )
    
    
    # Debugging message checkpoint
    if debug:
        feedbacklines = ['MAKE_BPMASK.DEBUG        Parameters set manually or determined from config file:',
                         'MAKE_BPMASK.DEBUG            {0: >16} : {1}'.format( 'dark_file', dark_file ),
                         'MAKE_BPMASK.DEBUG            {0: >16} : {1}'.format( 'bp_threshold', bp_threshold ),
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
    dark_data_mean = fits.getdata( dark_file, 0 )
    
    
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
    hdu.header['DARKFILE'] = ( dark_file.split('/')[-1], 'Dark file used' )
    
    
    # Copies some header values from the mean dark file used
    with fits.open( dark_file ) as dark_hdu:
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
    hdu.writeto( outfile )
    
    
    
        
