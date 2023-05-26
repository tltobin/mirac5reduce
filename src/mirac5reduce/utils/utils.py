################## Importing packages ####################

from glob import glob
import os
from astropy.io import fits


################## Functions ####################


def get_raw_filenames( raw_name_fmt, startno, endno, raw_file_path  ):
    """
    Simple utility function to get a sorted list of the raw data files from a starting file number (startno)
    to and including the end file number (endno).
    
    It will also print a brief warning if the number of file names retrieved is not what was expected from the
    start and end file numbers.
    
    Note from TLT 5/25/2023: As I don't know if the file name format will have the fixed width file numbers 
    or not, this routine is meant to be robust in determining that from the files present in the 
    raw_file_path.
    
    Required Parameters
    -------------------
    
            raw_name_fmt    String containing an asterisk
            
                                Naming convention used for the raw fits files created by MIRAC5. Should be 
                                formatted such that the file number, that changes from file to file, is 
                                represented by a '*'.
                                
                                Eg. if file names are like 'gsnapImg201.fits', this should be set to 
                                'gsnapImg*.fits'
                                
                                Same as the parameter of the same name in the init file.
            
            startno         Integer
                                
                                File number of the first file to retrieve the name of.
            
            endno           Integer
                                
                                File number of the last file (inclusive) to retrieve the name of.
                                
            raw_file_path   String
                                
                                Path where the raw fits files from the telescope are stored.
    
    Returns
    -------
    
            filelist        List of Strings
                            
                                Sorted list of file names with the desired format in the raw_file_path 
                                directory with file numbers from startno to endno, inclusive.
                                
                                File names do not include the file path.
    """
    
    # Checks if there are any files in the raw_file_path that have a file number section starting with '0'
    # If there are none, can use the file numbers directly through glob
    checkfiles = glob( os.path.join( raw_file_path,  raw_name_fmt.replace( '*', '0*' ) ) )
    if len( checkfiles ) == 0:
        
        # Creates the python formatting template to use
        fname_template = raw_name_fmt.replace( '*', '{0}' )
    
    # If some of the file numbers do start with 0, will need to determine the width of the fixed-width file
    #   numbering section of the file name
    else:
        
        # Finds the part of the file name retrieved in checkfiles that is the file number and finds its width
        #   Assumes they are all the same
        testfile = checkfiles[0].replace( '{0}/'.format(raw_file_path), '' )
        raw_name_str, raw_name_end = raw_name_fmt.split('*')
        test_fno = testfile.replace( raw_name_str, '' ).replace( raw_name_end, '' )
        ndigits = len(test_fno)
        
        # Creates the python formatting template to use
        fname_template = raw_name_fmt.replace( '*', '{0:0>'+str(ndigits)+'}' )
    
    # Generates expected file name list from file numbers
    filenames = [ fname_template.format(i) for i in range( startno, endno+1 ) ]
    
    # Prunes this down to just files that actually exist in the raw file path
    filelist = [ fname for fname in filenames if os.path.isfile( os.path.join( raw_file_path, fname ) ) ]
    
    # Checks if all expected files were found; if not, prints warning
    if len(filelist) < len(filenames):
        print('Warning: Missing {0} expected files between file numbers {1} and {2}.'.format( \
                                                            len(filenames)-len(filelist), startno, endno ))
    
    
    # Returns list of files that were found
    return filelist


def write_mean_frame( meanfile_name, avgframe, frametype, raw_filelist, raw_filepath = None ):
    """
    Saves mean frame calculated from a list of raw frames to an output fits file, populating the header with
    some calculation details and some keys copied over from the first raw fits file used to calculate it.
    
    Required Parameters
    -------------------
    
            meanfile_name   String
            
                                The file name(s) (with paths) to which the output fits file with the mean
                                frame and associated data will be saved.
                                
            avgframe        NumPy Array
                            
                                The mean frame calculated from the provided input frames. Will be saved in the
                                output fits file's 0th extension.
            
            frametype       String: 'Dark', 'Flat', 'Obs', etc.
                                
                                Image type (i.e. dark, flat, obs) of the data frames averaged to create the 
                                avgframe. Used to set the 'FILETYPE' fits header key.
            
            raw_filelist    List of Strings
                            
                                Sorted list of file names used to calculate the avgframe provided.
                                
                                
                            
    Optional Parameters
    -------------------
            
            raw_filepath    String or None
                                
                                [ Default = None ]
                            
                                The path where the raw fits files in raw_filelist are stored. 
                                
                                If provided (not None), will copy over a number of header key cards from the
                                first fits file in raw_filelist into the new output file.

                            
    Output Files Generated
    ----------------------
    
        [meanfile_name]
        
                            Fits file containing (in extension 0) the provided avgframe as data. 
                            
                            Copies some info from original fits file headers to the extension 0 header of this
                            file, as well as saving the start and end file numbers and the total number of
                            files used.
    
    """
    
    
    
    # Creates hdu to save to file with mean frame
    hdu = fits.PrimaryHDU( avgframe )
    
    # Populates header directly with general info 
    hdu.header['FILETYPE'] =   'Combined {0}'.format( frametype.capitalize() )
    hdu.header['NFRAMES' ] = ( len(raw_filelist), 'Number raw frames used' )
    hdu.header['FILE_STR'] = ( raw_filelist[0]  , 'First raw file used' )
    hdu.header['FILE_END'] = ( raw_filelist[-1] , 'Last raw file used' )
    hdu.header['COMBTYPE'] = ( 'MEAN'       , 'How raw frames were combined' )
    
    # If a raw_filepath was provided, checks first file for desired keys and copies any to header of output
    if raw_filepath is not None:
        
        # List of header keys to copy over from the first raw dark file
        keys_to_copy = [ 'DATE', 'TIMEDAY', 'PLUS',                         # when first dark frame was taken
                         'SNAP_VER', 'SNAPDATE', 'DEVICE', 'PARTNUM',       # versioning, if ever wanted
                         'DCFILE', 'INITFILE',                              # ref files of potential interest
                         'WINTRANS', 'DETPITCH', 'APERDIST', 'APERDIAM',    # some info about exposures, if wanted
                         'FRMRATE', 'INTEGRT', 'INTEGRTM',                  # frame rate and integration
                         'GAIN_SET', 'CH0POWER', 'CH1POWER', 'CH2POWER', 'CH3POWER', 'CH4POWER', 'CH5POWER' ]

        # Opens the first raw dark file used to create the mean and copies some values from its header to the
        #   new header. Assumes these are in the 0th extension, not the data ext
        with fits.open( os.path.join( raw_filepath, raw_filelist[0] ) ) as raw_ref_hdu:
            for key in keys_to_copy:
                if key in raw_ref_hdu[0].header.keys():
                    hdu.header[key] = raw_ref_hdu[0].header.cards[key][1:]
    
    # Finally, write this hdu to the output file
    hdu.writeto( meanfile_name )