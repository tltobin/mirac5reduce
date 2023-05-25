##  Adapted from Rory Bowen's script for the MIRAC-5 Geosnap by Taylor Tobin

################## Importing packages ####################

import numpy as np


################# Numerical precision ###################

__epsilon = np.finfo(float).eps


################## Functions ####################



def medabsdev(data, axis=None, keepdims=False, nan=True):
    """
    Median Absolute Deviation
    
    A "robust" version of standard deviation. Runtime is the same as `astropy.stats.funcs.mad_std`.
    
    Adapted from Rory Bowen's script for analyzing the MIRAC-5 Geosnap 1/f noise.
    
    Required Parameters
    -------------------
    
            data        NumPy array
            
                            The input data.
                            
    Optional Parameters
    -------------------
    
            axis        Int, tuple of ints, or None
            
                            [ Default = None ]
                            
                            Axis or axes along which the deviation is computed. The default is to compute the 
                            deviation of the flattened array.
        
                            If this is a tuple of ints, a standard deviation is performed over  multiple axes, 
                            instead of a single axis or all the axes as before. This is the equivalent of 
                            reshaping the input data and then taking the standard deviation.
                            
            keepdims    Bool
                            
                            [ Default = False ]
                            
                            If this is set to True, the axes which are reduced are left in the result as 
                            dimensions with size one. With this option, the result will broadcast correctly 
                            against the original `arr`.
                            
            nan         Bool
            
                            [ Default = True ]
                            
                            Ignore NaNs? Default is True.
                            
    Returns
    -------
    
            sigma       NumPy Array or Float
                            
                            The value(s) of the median absolute deviation calculated along the desired axes
                            
    """
    medfunc = np.nanmedian if nan else np.median
    meanfunc = np.nanmean if nan else np.mean

    if (axis is None) and (keepdims==False):
        data = data.ravel()
    
    # Scale factor to return result equivalent to standard deviation.
    sig_scale = 0.6744897501960817
    
    med = medfunc(data, axis=axis, keepdims=True)
    absdiff = np.abs(data - med)
    sigma = medfunc(absdiff, axis=axis, keepdims=True)  / sig_scale
    
    # Check if anything is near 0.0 (below machine precision)
    mask = sigma < __epsilon
    if np.any(mask):
        sigma[mask] = (meanfunc(absdiff, axis=axis, keepdims=True))[mask] / 0.8
    mask = sigma < __epsilon
    if np.any(mask):
        sigma[mask] = 0.0
        
        
    if len(sigma)==1:
        return sigma[0]
    elif not keepdims:
        return np.squeeze(sigma)
    else:
        return sigma
