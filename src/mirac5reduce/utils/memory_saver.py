################## Importing packages ####################

import numpy as np
from astropy.io import fits
from math import ceil

################## Functions ####################


def lm_mean_frame( filenames, ext = None, maxframes = 200, logfile = None ):
    """
    Calculates the mean frame of data read in from one or more fits files in such a way that prevents all of
    the input frames from being loaded into the memory simultaneously.
    
    To combine 2D frames located in a known extension of multiple files, provide the list of file names as
    filenames and the fits extension where the desired arrays are stored in each file as ext.
    
    To combine 2D frames located in different extensions of a single fits file, provide the name of that
    fits file as a single string as filenames and set ext to None. (Extensions that have no data associated
    with them will be ignored.)
    
    Required Parameters
    -------------------
    
            filenames       String or List of Strings
            
                                The file name(s) (with paths) where the data arrays to be combined are stored.
                            
                                If a single string is provided, assumes frames are stored in separate 
                                extensions of that single indicated fits file.
                            
                                If a list of strings is provided, assumes frames are stored in the extension 
                                specified by ext for all fits files specified.
    
    Optional Parameters
    -------------------
    
            ext             Int or None
            
                                [ Default = None ]
                            
                                The extension in each fits file specified in filenames where the data frames 
                                to be combined are stored. Is only used if multiple files are indicated by 
                                filenames.
                            
            maxframes       Int
                            
                                [ Default = 200 ]
                            
                                The maximum number of input 2D frames that are allowed to be loaded into 
                                memory at any given time. Keep in mind that an additional 2D frame of the same 
                                size with the combined results will also be present in memory.
                            
            logfile         String or None
                            
                                [ Default = None ]
                            
                                File name (and path) of a log file in which to provide feedback on the 
                                function's progress. If not provided, progress will be printed to the 
                                terminal.
                            
    Returns
    -------
    
            avgframe        2D NumPy Array
                            
                                The mean frame calculated from the provided input frames.
    """
    
    # Initializes bool switch to say how frames were provided. If 1, frames are each in separate files.
    #   If 0, frames are in different extensions of the same file.
    sepfiles = 1
    
    # Sets switch to 0 if single file provided. If filenames provided as string, changes to list.
    if isinstance(filenames, list) and len(filenames)==1:
        sepfiles = 0
    if isinstance( filenames, str):
        sepfiles = 0
        filenames = [ filenames, ]
        
    # Retrieves the shape of a single 2D frame
    with fits.open( filenames[0], mode='readonly' ) as hdulist:
        
        # If each frame in its own file, just looks at the indicated extension
        #   While we're in the if statement, also saves total number of frames to be combined and generates
        #   extlist that is just the extension index for each file (all the same)
        if sepfiles == 1:
            ext0 = ext
            totframes = len( filenames )
            extlist = [ ext, ] * totframes
        
        # If frames are stored in different extensions of this file, gets a list of extension indices within
        #   that file that have 2D data 
        else:
            extlist = [ i for i in range(len( hdulist )) if ( 'NAXIS' in hdulist[i].header.keys() and hdulist[i].header['NAXIS'] == 2 ) ]
            ext0 = extlist[0]
            totframes = len( extlist )
            
            # While here, makes filenames a list of the same file name with the same length as the extlist
            filenames = [ filenames[0], ] * len(extlist)
            
        # Retrieves the shape of the 2D data in that extension
        frame_shape = hdulist[ext0].data.shape
        
    # Creates an empty array to build up with the cumulative average and later return
    avgframe = np.zeros( frame_shape )
    
    
    # Creates arrays of number of frames per chunk and the associated weights to use for each chunk of frames 
    #   when added to avgframe array 
    nchunks = ceil( totframes / maxframes )
    nframes = np.array( [ maxframes, ]*nchunks )
    if ( totframes % maxframes ) != 0:
        nframes[-1] = ( totframes % maxframes )
    frameweights = nframes / totframes
    
    # Before starting, prints some feedback to log or terminal
    feedbacklines = [        'LM_MEAN_FRAME:       Calculating memory-saving mean frame:',
                             '                         {0} Frames of shape {1}'.format( totframes, frame_shape ),
                             '                         Max {0} frames loaded simultaneously ({1} chunks)'.format(maxframes, nchunks), ]
    if np.sum( frameweights ) != 1.0:
        tmpsum = np.sum( frameweights )
        feedbacklines.append('                     Warning: Chunk weights do not sum to 1. Actual Sum: {0} (diff {1:.2e})'.format( tmpsum, 1.-tmpsum )  )
    feedbacklines.append(    '                     Calculating average frame' )
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write( '\n' )
            lf.write( '\n'.join( feedbacklines ) )
    else:
        for i in range(len(feedbacklines)-1):
            print( feedbacklines[i] )
        print( feedbacklines[-1], end='' )
    
    # Actually iterates through frames, reading them in by chunks and building up the avgframe
    # In both use cases, should have list of filenames and extlist with one entry per frame, even if 
    #   duplicates, so don't need to separate by sepframes switch
    for i, nframes_in_chunk in enumerate( nframes ):
        
        # Adds to feedback one period per chunk to track progress
        if logfile is not None:
            with open(logfile,'a') as lf:
                lf.write('.')
        else:
            print('.', end='')
        
        # Determines the indices of the files in filenames that will be read in for that chunk of frames
        file_idx_str = i * maxframes
        file_idx_end = file_idx_str + nframes_in_chunk
        
        # Retrieves the data from those files and builds them into a list, which will be turned into an 
        #   array with frame as the 0th axis
        chunk_frames = list()
        for j in range( file_idx_str, file_idx_end ):
            chunk_frames.append( fits.getdata( filenames[j], extlist[j], header = False ) )
        chunk_frames = np.array( chunk_frames )
        
        # Then calculate the mean frame, weight it, and add it to the avgframe
        mean_chunk = np.nanmean( chunk_frames, axis=0 )
        avgframe += frameweights[i] * mean_chunk
    
    # Tidies up feedback lines
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write('Done.\n')
    else:
        print('Done.')
    
    
    # Returns final average frame
    return avgframe
            
            
        
    