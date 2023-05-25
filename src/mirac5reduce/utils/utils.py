################## Importing packages ####################

from glob import glob
import os


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
    