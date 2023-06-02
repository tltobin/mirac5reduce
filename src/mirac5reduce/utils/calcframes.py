################## Importing packages ####################

from .. import __version__

import numpy as np
from astropy.io import fits
from math import ceil
from collections import OrderedDict

################## Functions ####################


def calc_mean_frame( filenames, ext = None, maxframes = 200, logfile = None ):
    """
    Calculates the mean frame of data read in from one or more fits files.
    
    To combine 2D frames located in a known extension of multiple files, provide the list of file names as
    filenames and the fits extension where the desired arrays are stored in each file as ext.
    
    To combine 2D frames located in different extensions of a single fits file, provide the name of that
    fits file as a single string as filenames and set ext to None. (Extensions that have no data associated
    with them will be ignored.)
    
    Has optional memory-saving capabilities, to place limits on the number of frames that can be loaded into
    memory simultaneously.
    
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
                            
            maxframes       Int or None
                            
                                [ Default = 200 ]
                            
                                The maximum number of input 2D frames that are allowed to be loaded into 
                                memory at any given time. Keep in mind that an additional 2D frame of the same 
                                size with the combined results will also be present in memory.
                                
                                If set to None, will operate with no limit, and read all provided data arrays
                                in to memory simultaneously.
                            
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
    
    # If max number of frames was provided (as not None), determines how it needs to be split up into chunks
    if maxframes is not None:
    
        # Creates arrays of number of frames per chunk and the associated weights to use for each chunk of frames 
        #   when added to avgframe array 
        nchunks = ceil( totframes / maxframes )
        nframes = np.array( [ maxframes, ]*nchunks )
        if ( totframes % maxframes ) != 0:
            nframes[-1] = ( totframes % maxframes )
        chunkweights = nframes / totframes
        loopframes = maxframes
    
        # Before starting, prints some feedback to log or terminal
        feedbacklines = [        'CALC_MEAN_FRAME:     Calculating mean frame:',
                                 '                         {0} Frames of shape {1}'.format( totframes, frame_shape ),
                                 '                         Max {0} frames loaded simultaneously ({1} chunks)'.format(maxframes, nchunks) ]
        if np.sum( chunkweights ) != 1.0:
            tmpsum = np.sum( chunkweights )
            feedbacklines.append('                     Warning: Chunk weights do not sum to 1. Actual Sum: {0} (diff {1:.2e})'.format( tmpsum, 1.-tmpsum )  )
        feedbacklines.append(    '                     Calculating average frame' )
        if logfile is not None:
            with open(logfile,'a') as lf:
                lf.write( '\n'.join( feedbacklines ) )
        else:
            for i in range(len(feedbacklines)-1):
                print( feedbacklines[i] )
            print( feedbacklines[-1], end='' )
    
    # If there is no limit on number of frames that can be read in, just has single chunk with all frames
    else:
    
        # Creates same variables as memory-saving version 
        nchunks = 1
        nframes = np.array([ totframes, ])
        chunkweights = nframes / totframes
        loopframes = 0
        
        # Before starting, prints some feedback to log or terminal
        feedbacklines = [        'CALC_MEAN_FRAME:     Calculating mean frame:',
                                 '                         {0} Frames of shape {1}'.format( totframes, frame_shape ),
                                 '                     Calculating average frame...' , ]
        if logfile is not None:
            with open(logfile,'a') as lf:
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
        file_idx_str = i * loopframes
        file_idx_end = file_idx_str + nframes_in_chunk
        
        # Retrieves the data from those files and builds them into a list, which will be turned into an 
        #   array with frame as the 0th axis
        chunk_frames = list()
        for j in range( file_idx_str, file_idx_end ):
            chunk_frames.append( fits.getdata( filenames[j], extlist[j], header = False ) )
        chunk_frames = np.array( chunk_frames )
        
        # Then calculate the mean frame, weight it, and add it to the avgframe
        mean_chunk = np.nanmean( chunk_frames, axis=0 )
        avgframe += chunkweights[i] * mean_chunk
    
    # Tidies up feedback lines
    if logfile is not None:
        with open(logfile,'a') as lf:
            lf.write('Done.\n')
    else:
        print('Done.')
    
    
    # Returns final average frame
    return avgframe
            
            
def calc_chopnod_frame( filenames, ext = None, chopfreq = None, nodfreq = None,
                        maxframes = 200, logfile = None, _fitsdict_ = False ):
    """
    Calculates the average chop-nod difference frame of data read in from one or more fits files.
    
    To combine 2D frames located in a known extension of multiple files, provide the list of file names as
    filenames and the fits extension where the desired arrays are stored in each file as ext.
    
    To combine 2D frames located in different extensions of a single fits file, provide the name of that
    fits file as a single string as filenames and set ext to None. (Extensions that have no data associated
    with them will be ignored.)
    
    Has optional memory-saving capabilities, to place limits on the number of frames that can be loaded into
    memory simultaneously.
    
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
            
            chopfreq        Float or None
                                
                                [ Default = None ]
                                
                                Frequency (in Hz) at which chopping occurs. This is converted into number of 
                                frame files using header keys in the raw data files.
                                
                                Chopfreq should be larger than nodfreq.
                                
                                If set to None, assumes no chops occur in the data set.
                                
            nodfreq         Float or None
                                
                                [ Default = None ]
                                
                                Frequency (in Hz) at which nodding occurs. This is converted into number of 
                                frame files using header keys in the raw data files.
                                
                                If set to None, assumes no nods occur in the data set.
            
            maxframes       Int or None
                            
                                [ Default = 200 ]
                            
                                The maximum number of input 2D frames that are allowed to be loaded into 
                                memory at any given time. Keep in mind that an additional 2D frame of the same 
                                size with the combined results will also be present in memory.
                                
                                If set to None, will operate with no limit, and read all provided data arrays
                                in to memory simultaneously.
                            
            logfile         String or None
                            
                                [ Default = None ]
                            
                                File name (and path) of a log file in which to provide feedback on the 
                                function's progress. If not provided, progress will be printed to the 
                                terminal.
            
            _fitsdict_      Boolean
                                
                                [ Default = False ]
                                
                                If set to True, will also return a dictionary with values calculated by
                                this function, to be saved to the header of the output fits file by
                                the utils.write_chopnod_frame function.
                            
    Returns
    -------
    
            diffframe       2D NumPy Array
                            
                                The mean chop-nod difference frame calculated from the provided input frames.
            
            
        (Optional)
            
            header_dict     OrderedDict
                                
                                Only if _fitsdict_ = True. Dictionary containing several calculated values
                                that can be provided to utils.write_chopnod_frame to be saved in the created
                                fits file's header.
                                
                                
    """
    
    # Only calc if chopfreq or nodfreq is not None
    if (chopfreq is not None) or (nodfreq is not None):
    
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
            
            # Also retrieves frame rate from header, in frames per second
            framerate = hdulist[0].header['FRMRATE']
    
        
        # Determines the number of frames in each chop cycle and position
        chop_cycle_frames = round( float(framerate) / float(chopfreq) )     # frames / chopcycle = (frames/sec) / (chopcycle/sec)
        chop_pos_frames   = round( chop_cycle_frames / 2 )                  # 2 chop positions per cycle
        
        # Determines the number of frames in each nod cycle and position
        if nodfreq is not None:
            nod_cycle_frames = round( float(framerate) / float(nodfreq) )   # frames / nodcycle = (frames/sec) / (nodcycle/sec)
            nod_pos_frames   = round( nod_cycle_frames / 2 )                # 2 nod positions per cycle
        else:
            nod_pos_frames = totframes
            nod_cycle_frames = nod_pos_frames * 2
        
        # Number of chop/nod cycles
        Nchopcycles = int(totframes/chop_cycle_frames)
        Nnodcycles  = int(totframes/nod_cycle_frames )
        
        # Number of chop cycles per nod *position*
        Nchopcyc_per_nodpos = int( nod_pos_frames / nod_cycle_frames )
        
        # Creates array of +/- 1 indicating whether each frame will be added or subtracted from the total
        single_chopcycle_signs = np.concatenate(( np.ones(chop_pos_frames), -1*np.ones(chop_pos_frames) ))
        
        if totframes >= chop_cycle_frames :
            chopsigns = np.concatenate( [single_chopcycle_signs,] * Nchopcycles )
        else:
            chopsigns = single_chopcycle_signs[:totframes]
        
        single_nodcycle_signs = np.concatenate(( np.ones(nod_pos_frames), -1*np.ones(nod_pos_frames) ))
        
        if totframes >= nod_cycle_frames:
            nodsigns = np.concatenate( [single_nodcycle_signs,] * Nnodcycles )
        else:
            nodsigns = single_nodcycle_signs[:totframes]
        
        framesigns = chopsigns * nodsigns
        
        # And calculates the weight applied to each frame to make the result an average
        frameweight = 1. / totframes
        
        
        
        
    
        # If max number of frames was provided (as not None), determines how it needs to be split up into chunks
        if maxframes is not None:
    
            # Creates arrays of number of frames per chunk and the associated weights to use for each chunk of frames 
            #   when added to avgframe array 
            nchunks = ceil( totframes / maxframes )
            nframes_per_chunk = np.array( [ maxframes, ]*nchunks )
            if ( totframes % maxframes ) != 0:
                nframes_per_chunk[-1] = ( totframes % maxframes )
            chunkweights = nframes_per_chunk * frameweight
            loopframes = maxframes
    
            # Before starting, prints some feedback to log or terminal
            feedbacklines = [        'CALC_CHOPNOD_FRAME:  Calculating mean chop/nod difference frame:',
                                     '                         {0} Frames of shape {1}'.format( totframes, frame_shape ),
                                     '                         Max {0} frames loaded simultaneously ({1} chunks)'.format(maxframes, nchunks) ]
            if np.sum( chunkweights ) != 1.0:
                tmpsum = np.sum( chunkweights )
                feedbacklines.append('                     Warning: Chunk weights do not sum to 1. Actual Sum: {0} (diff {1:.2e})'.format( tmpsum, 1.-tmpsum )  )
            feedbacklines.append(    '                     Calculating average difference frame' )
            if logfile is not None:
                with open(logfile,'a') as lf:
                    lf.write( '\n'.join( feedbacklines ) )
            else:
                for i in range(len(feedbacklines)-1):
                    print( feedbacklines[i] )
                print( feedbacklines[-1], end='' )
    
        # If there is no limit on number of frames that can be read in, just has single chunk with all frames
        else:
    
            # Creates same variables as memory-saving version 
            nchunks = 1
            nframes_per_chunk = np.array([ totframes, ])
            chunkweights = nframes_per_chunk * frameweight
            loopframes = 0
        
            # Before starting, prints some feedback to log or terminal
            feedbacklines = [        'CALC_CHOPNOD_FRAME:  Calculating mean chop/nod difference frame:',
                                     '                         {0} Frames of shape {1}'.format( totframes, frame_shape ),
                                     '                     Calculating average difference frame...' , ]
            if logfile is not None:
                with open(logfile,'a') as lf:
                    lf.write( '\n'.join( feedbacklines ) )
            else:
                for i in range(len(feedbacklines)-1):
                    print( feedbacklines[i] )
                print( feedbacklines[-1], end='' )
        
        
        
        # Creates an empty array to build up with the cumulative mean difference and later return
        diffframe = np.zeros( frame_shape )
        
        
        # Actually iterates through frames, reading them in by chunks and building up the diffframe
        # In both use cases, should have list of filenames and extlist with one entry per frame, even if 
        #   duplicates, so don't need to separate by sepframes switch
        for i, nframes_in_chunk in enumerate( nframes_per_chunk ):
        
            # Adds to feedback one period per chunk to track progress
            if logfile is not None:
                with open(logfile,'a') as lf:
                    lf.write('.')
            else:
                print('.', end='')
        
            # Determines the indices of the files in filenames that will be read in for that chunk of frames
            file_idx_str = i * loopframes
            file_idx_end = file_idx_str + nframes_in_chunk
        
            # Retrieves the data from those files and builds them into a list, which will be turned into an 
            #   array with frame as the 0th axis, already multiplied by +1/-1 from framesigns array
            chunk_frames = list()
            for j in range( file_idx_str, file_idx_end ):
                chunk_frames.append( fits.getdata( filenames[j], extlist[j], header = False ) * framesigns[j] )
            chunk_frames = np.array( chunk_frames )
        
            # Then calculate the mean frame, weight it, and add it to the avgframe
            mean_chunk = np.nanmean( chunk_frames, axis=0 )
            diffframe += chunkweights[i] * mean_chunk
        
    
        # Tidies up feedback lines
        if logfile is not None:
            with open(logfile,'a') as lf:
                lf.write('Done.\n')
        else:
            print('Done.')
        
    
    
    # Otherwise, just use calc_mean_frame
    else:
        
        # Before starting, prints some feedback to log or terminal
        feedbacklines = [        'CALC_CHOPNOD_FRAME:  No chops or nods detected for chop/nod differencing.',
                                 '                         Calculating average frame with calc_mean_frame.' ]
        if logfile is not None:
            with open(logfile,'a') as lf:
                for i in range(len(feedbacklines)):
                    lf.write( '{0}\n'.format(feedbacklines[i]) )
        else:
            for i in range(len(feedbacklines)):
                print( feedbacklines[i] )
        diffframe = calc_mean_frame( filenames, ext = ext, maxframes = maxframes, logfile = logfile )
    
    
    
    # If returning the header_dict, creates it
    if _fitsdict_:
        header_dict = OrderedDict({ 'CHOPFREQ'   : ( chopfreq           , 'chop frequency (Hz)'                    ),
                                    'CHOPFRAM'   : ( chop_pos_frames    , 'frames per chop position'               ),
                                    'CHOPCYCL'   : ( Nchopcycles        , 'number of chop (AB) cycles'             ),
                                    'NODFREQ'    : ( nodfreq            , 'nod frequency (Hz)'                     ),
                                    'NODFRAM'    : ( nod_pos_frames     , 'frames per nod position'                ),
                                    'NODCYCL'    : ( Nnodcycles         , 'number of nod (12) cycles'              ),
                                    'CSPERNOD'   : ( Nchopcyc_per_nodpos, 'number of chop cycles per nod position' )
                                    })
        return diffframe, header_dict
    
    else:
        return diffframe
        
    