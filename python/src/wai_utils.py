"""
Created on Fri Feb 14 2025

@author: Jonathan Bitton
"""
# Standard library imports
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Third-party imports
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from pandas.arrays import DatetimeArray

# Local imports
import data_utils

def coefvalues(
    wave: np.ndarray,
    periods: np.ndarray,
    ts: Union[int, float, timedelta],
    COI: Optional[np.ndarray] = None,
    pstr: Optional[Union[str, List[str]]] = None,
    pval: Optional[Union[int, float, ArrayLike, pd.Series, pd.DataFrame]] = None,
    pspread: Optional[Union[int, float, ArrayLike, pd.Series, pd.DataFrame]] = None,
    units: str = 'DAYS'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Union[int, float]]]:
    """
    Extract mean periodic bands from wavelet transform coefficients.
    
    This function computes the mean wavelet coefficients and power values for specified
    periods, both with and without considering the cone of influence (COI).
    
    Parameters
    ----------
    wave : np.ndarray
        Wavelet transform coefficients array with shape (n_scales, n_times)
    power : np.ndarray
        Power spectrum (squared amplitude) of wavelet coefficients with shape (n_scales, n_times)
    periods : np.ndarray
        Array of periods corresponding to each scale
    ts : int, float or timedelta
        Sampling period, either as a numerical value or as a timedelta
    COI : np.ndarray, optional (default=None)
        Cone of influence array indicating areas affected by edge effects.
        If None, no edge effects are considered
    pstr : str or list of str, optional (default=None)
        (List of) period identifier(s). 
        If given in format: letter (among 'y' for years, 'm' for months, 'w' for weeks, 'd' for days, 'h' for hours) 
        followed by a number, pval and pspread are automatically computed (if not provided)
        Default values: ["y1", "m8", "m7", "m6", "m5", "m4", "m3", "m2", "m1", "w3", "w2", "w1", "d1", "h12"] 
        for 1 year, [8:1] months, [3:1] weeks, 1 day, 12 hours
    pval : int, float or np.ndarray, optional (default=None)
        Array of values corresponding to each period in pstr given in units of x-axis (e.g. time or space)
        If None, computed based on pstr (if correctly formatted), assumed to be time data in days
        Default: 365*number for years, 30*number for months, 7*number for weeks, number for days, number/24 for hours
    pspread : int, float or np.ndarray, optional (default=None)
        Array of half-spreads (in days) to use for computing means
        If None, computed based on pval, assumed to be in days
        Default: 30 for years, 15 for months, 3.5 for weeks, 0.3 for days, 0.1 for hours
    Units : str, optional (default=days)
        Units for timedelta conversion when periods and COI are durations
    
    Returns
    -------
    str2val : dict
        Dictionary mapping period identifiers (pstr) to their x-axis values (pval)
    coef : pd.DataFrame
        Mean power values for all coefficients
    sgn : pd.DataFrame
        Mean complex wavelet coefficients for all values
    idx : pd.DataFrame
        Boolean mask indicating which coefficients were used for each period
    coefcoi : pd.DataFrame, optional
        Mean power values with border-affected values removed (using COI)
    sgncoi : pd.DataFrame, optional
        Mean complex wavelet coefficients with border-affected values removed    
    
    Examples
    --------
    >>> # Extract default periodic bands from wavelet transform results
    >>> coefcoi, coef, idx, sgncoi, sgn, str2val = coefvalues(wave, power, periods, COI, 1.0)
    
    >>> # Extract specific periods with custom spreads
    >>> pstr = ["y1", "m6", "m3"]
    >>> pval = np.array([365, 180, 90])
    >>> pspread = np.array([30, 15, 15])
    >>> coefcoi, coef, idx, sgncoi, sgn, str2val = coefvalues(wave, power, periods, COI, 1.0, 
                                                              pstr, pval, pspread)
    
    Notes
    -----
    - When pval is not provided, the function automatically computes appropriate
      values based on the period identifiers in pstr.
    - The function handles both numeric and datetime inputs for periods.
    - Edge effects are handled by masking coefficients within the COI.
    - If COI is not provided, coefcoi and sgncoi will be None.
    """
    # Defining periods to analyze
    if pstr is None:  # Default (automatic periods, x-axis locations and spreads)
        if pval is not None:
            raise ValueError('Missing identifier (pstr) for positions in pval')
        # Default periods to extract
        pstr = ["y1", "m8", "m7", "m6", "m5", "m4", "m3", "m2", "m1", "w3", "w2", "w1", "d1", "h12"]
        pval = np.array([365, 240, 210, 180, 150, 120, 90, 60, 30, 21, 14, 7, 1, 0.5])
        pspread = np.array([30, 15, 15, 15, 15, 15, 15, 15, 15, 3.5, 3.5, 3.5, 0.3, 0.1])
        nper = len(pstr)  # Number of periods to extract
    elif pval is not None:  # User-defined periods and x-axis locations (and spreads)
        # Ensure pstr is iterable
        if isinstance(pstr, str):
            pstr = [pstr]   
        # Validate and standardize inputs
        pval, pspread = verify_inputs(pstr, pval, pspread)
        # Inform user about assumptions for timedelta objects
        if isinstance(periods[0], timedelta) and (units == 'DAYS'):
            import warnings
            warnings.warn('Assumed pval and pspread represent days. To modify this option, '
                          'specify the correct units using the "units" argument', UserWarning)
        # Sort in descending order
        pstr, pval, pspread = psort(pstr, pval, pspread)
    else:  # User-defined periods & automatic x-axis locations and spreads
        import re
        # Ensure pstr is iterable
        if isinstance(pstr, str):
            pstr = [pstr]
        
        # Preallocate arrays
        nper = len(pstr)
        pval = np.empty(nper, dtype=float)
        pspread = np.empty(nper, dtype=float)
        
        # Define a dictionary for unit conversion and default spreads
        unit_to_days = {'y': 365, 'm': 30, 'w': 7, 'd': 1, 'h': 1/24}
        default_spreads = {'y': 30, 'm': 15, 'w': 3.5, 'd': 0.3, 'h': 0.1}

        # Parse period identifiers and calculate corresponding dates and spreads
        for j, period in enumerate(pstr):
            # Extract unit and number from period identifier
            match = re.match(r'([ymwdh])(\d+)', period)
            if not match:
                raise ValueError(
                    f"Period {period} not recognized. Use a letter ('y', 'm', 'w', 'd', 'h') "
                    f"followed by a number. Example: 'y1' for 1 year."
                )
            unit, number = match.groups()
            number = float(number)
        
            # Compute pval (convert to days)
            pval[j] = number * unit_to_days[unit]
        
            # Compute pspread based on conditions
            if unit == 'w' and number > 3:
                if number >= 52:
                    pspread[j] = 30  # Convert spread to years for weeks >= 52
                else:
                    pspread[j] = 15  # Convert spread to months for weeks > 3
            elif unit == 'd' and number >= 7:
                if number >= 365: 
                    pspread[j] = 30  # Convert spread to years for days >= 365
                elif number >= 30:
                    pspread[j] = 15  # Convert spread to months for days >= 30
                else:
                    pspread[j] = 3.5  # Convert spread to weeks for days >= 7
            elif unit == 'h' and number >=24:
                if number >= 8760:
                    pspread[j] = 30  # Convert spread to years for hours >= 8760
                if number >= 720:
                    pspread[j] = 15  # Convert spread to months for hours >= 720
                elif number >= 168:
                    pspread[j] = 3.5  # Convert spread to weeks for hours >= 168
                else:
                    pspread[j] = 0.3  # Convert spread to days for hours >= 24
            else:
                pspread[j] = default_spreads[unit]
        
        # Sort in descending order
        pstr, pval, pspread = psort(pstr, pval, pspread)
    
    # Power
    power = np.abs(wave)**2
    
    # Create dictionary mapping period strings to values
    str2val = {"units": units.lower()}
    str2val.update(zip(pstr, pval))
    
    # Check and convert Ts, periods and COI for timedelta objects
    if isinstance(ts, timedelta):
        periods, COI, ts = data_utils.convert_timdelta_array(units, periods, COI, ts)
        ts = ts.item()
    
    # Create a dataframe to store indices of coefficients for each period
    idx = pd.DataFrame(False, index=range(power.shape[0]), columns=pstr)
    
    # Define coefficients to keep for each period based on period bands
    v1 = pval + pspread  # Upper bound
    v2 = pval - pspread  # Lower bound
    pds = periods[:, np.newaxis]  # Reshape for broadcasting
    
    # Find coefficients within each period band
    idx.iloc[:, :] = (pds < v1) & (pds > v2)
    
    # Initialize output dataframes
    L = power.shape[1]
    coef = pd.DataFrame(np.nan, index=range(L), columns=pstr)
    sgn = pd.DataFrame(np.nan, index=range(L), columns=pstr, dtype=complex)
    
    if COI is not None:
        # Initialize COI-related output dataframes
        coefcoi = pd.DataFrame(np.nan, index=range(L), columns=pstr)
        sgncoi = pd.DataFrame(np.nan, index=range(L), columns=pstr, dtype=complex)
        
        # Identify edge-affected coefficients using the COI
        edgeaff = COI - pds
        edgeaff[edgeaff <= 0] = 10e6  # Replace non-positive values with large number
        b = np.nanargmin(edgeaff, axis=0)  # Find indices where edge effects begin
        b[b < 1e-10] = -1  # Handle edge cases
        
        # Create mask for COI (coefficients affected by edge effects)
        mask = np.ones_like(power, dtype=bool)  # Initialize mask as True
        rows = np.arange(power.shape[0])
        center_id = int(np.ceil(L / 2))
        
        # Left half of mask (first half of time series)
        left_mask = rows[:, np.newaxis] <= b[np.newaxis, :]
        mask[:, :center_id] = ~left_mask[:, :center_id]  # Set False for affected rows
        
        # Reflect for the right half (second half of time series)
        mask[:, center_id:] = np.fliplr(mask[:, :center_id])
        
        # Apply mask to power and wave arrays
        power2 = np.where(mask, np.nan, power)  # Keep only values outside COI
        wave2 = np.where(mask, np.nan, wave)
        
        # Compute means for each period
        for i in pstr:
            indices = idx[i]  # Boolean mask for this period
            if np.any(indices):
                # Mean wave/power values for all coefficients
                coef[i] = np.mean(power[indices, :], axis=0)
                sgn[i] = np.mean(wave[indices, :], axis=0)
                
                # Mean wave/power values excluding border-affected coefficients
                coefcoi[i] = np.mean(power2[indices, :], axis=0)
                sgncoi[i] = np.mean(wave2[indices, :], axis=0)
    
        return str2val, coef, sgn, idx, coefcoi, sgncoi
    else:
        # Compute means for each period
        for i in pstr:
            indices = idx[i]  # Boolean mask for this period
            if np.any(indices):
                # Mean wave/power values for all coefficients
                coef[i] = np.mean(power[indices, :], axis=0)
                sgn[i] = np.mean(wave[indices, :], axis=0)
                
        return str2val, coef, sgn, idx, None, None

def verify_inputs(
    pstr: List[str], 
    pval: Union[ArrayLike, pd.Series, pd.DataFrame, None], 
    pspread: Optional[Union[ArrayLike, pd.Series, pd.DataFrame]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and standardize input parameters for periods and spreads.
    
    This function checks that the input parameters have the correct dimensions,
    converts them to numpy arrays if needed, and initializes pspread to default
    values if not provided.
    
    Parameters
    ----------
    pstr : List[str]
        List of period identifiers (e.g., "y1", "m6", "d1")
    pval : array_like
        Array of values in x-axis units corresponding to each period in pstr.
    pspread : array_like, optional
        Array of half-spreads (in x-axis units) to use for computing means.
        If missing, default values will be calculated based on period values (assumed to be in days).
    
    Returns
    -------
    pval_array : np.ndarray
        Validated and standardized array of period values
    pspread_array : np.ndarray
        Validated and standardized array of spread values
    
    Raises
    ------
    ValueError
        If pval and pstr have different lengths
        If negative values are found in pval
    TypeError
        If pval is not a valid numeric array-like object
    
    Examples
    --------
    >>> pstr = ["y1", "m6", "d1"]
    >>> pval = [365, 180, 1]
    >>> pval_array, pspread_array = verify_inputs(pstr, pval)
    """
    # Convert pval to numpy array if it's not already
    if not isinstance(pval, np.ndarray):
        if isinstance(pval, pd.DataFrame):
            pval_array = pval.iloc[:, 0].to_numpy()
        else:
            try:
                if all(isinstance(x, int) for x in pval):
                    pval_array = np.array(pval, dtype=int)
                else:
                    pval_array = np.array(pval, dtype=float)
            except (ValueError, TypeError):
                raise TypeError("pval must be convertible to a numeric numpy array")
    else:
        pval_array = pval.copy()  # Create a copy to avoid modifying original
    
    # Check that pval and pstr have the same length
    if len(pval_array) != len(pstr):
        raise ValueError(f"Length mismatch: len(pstr)={len(pstr)} but len(pval)={len(pval_array)}")
    
    # Check for negative values in pval
    if np.any(pval_array < 0):
        raise ValueError("Negative values found in pval. All periods must be positive.")
    
    # Initialize pspread if not provided
    if pspread is None:
        # Create default pspread values based on pval values
        pspread_array = np.empty_like(pval_array)
        
        # Issue a warning
        import warnings
        warnings.warn("pspread is not provided, default values will be used, assuming pval represents days.", UserWarning, stacklevel=2)
        
        # Apply different spread rules based on period magnitude
        for i, period in enumerate(pval_array):
            # Compute pspread based on conditions
            if period >= 365:  # Annual periods
                pspread_array[i] = 30
            elif period >= 30:  # Monthly periods
                pspread_array[i] = 15
            elif period >= 7:   # Weekly periods
                pspread_array[i] = 3.5
            elif period >= 1:   # Daily periods
                pspread_array[i] = 0.3
            else:               # Hourly periods
                pspread_array[i] = 0.1
    else:
        # Convert pspread to numpy array if it's not already
        if not isinstance(pspread, np.ndarray):
            if isinstance(pspread, (pd.DataFrame)):
                pspread_array = pspread.iloc[:, 0].to_numpy()
            else:
                try:
                    if all(isinstance(x, int) for x in pspread):
                        pspread_array = np.array(pspread, dtype=int)
                    else:
                        pspread_array = np.array(pspread, dtype=float)
                except (ValueError, TypeError):
                    raise TypeError("pspread must be convertible to a numeric numpy array")
        else:
            pspread_array = pspread.copy()  # Create a copy to avoid modifying original
        
        # Check that pspread and pstr have the same length
        if len(pspread_array) != len(pstr):
            raise ValueError(f"Length mismatch: len(pstr)={len(pstr)} but len(pspread)={len(pspread_array)}")
        
        # Check for negative values in pspread
        if np.any(pspread_array < 0):
            raise ValueError("Negative values found in pspread. All spreads must be positive.")
    
    return pval_array, pspread_array

def psort(
    pstr: List[str], 
    pval: np.ndarray, 
    pspread: np.ndarray
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Sort periods, locations, and spreads in descending order by period duration.
    
    Parameters
    ----------
    pstr : list of str
        List of period identifiers (e.g., "y1", "m6", "d1")
    pval : np.ndarray
        Array of values in x-axis units (e.g. time or space) corresponding to each period in pstr
    pspread : np.ndarray
        Array of half-spreads (in days) to use for computing means
    
    Returns
    -------
    sorted_pstr : list of str
        Sorted list of period identifiers
    sorted_pval : np.ndarray
        Sorted array of days corresponding to each period
    sorted_pspread : np.ndarray
        Sorted array of half-spreads
    
    Examples
    --------
    >>> pstr = ["d1", "m6", "y1"]
    >>> pval = np.array([1, 180, 365])
    >>> pspread = np.array([0.3, 15, 30])
    >>> sorted_pstr, sorted_pval, sorted_pspread = psort(pstr, pval, pspread)
    >>> sorted_pstr
    ['y1', 'm6', 'd1']
    """
    # Get the order of sorting (descending)
    idx = np.argsort(-pval)
    
    # Apply sorting
    sorted_pstr = [pstr[i] for i in idx]
    sorted_pval = pval[idx]
    sorted_pspread = pspread[idx]
    
    return sorted_pstr, sorted_pval, sorted_pspread

def extract_peaks(
    coef: pd.DataFrame,
    x_val: Union[ArrayLike, pd.Series, pd.DatetimeIndex, DatetimeArray],
    pstr: Union[str, List[str]],
    filter_val: Optional[Union[
        pd.DataFrame,
        Dict[str, Union[int, float, Sequence[Union[int, float]]]],
        Union[int, float, Sequence[Union[int, float]]]
    ]] = None,
    filter_per: Optional[str] = None,
    per: str = 'year',
    sgn: Optional[pd.DataFrame] = None,
    thresh: Optional[Union[int, float]] = -np.inf
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Extract repeating peaks in mean periodic bands.
    
    This function identifies peaks in wavelet coefficient data and applies filtering
    based on specified criteria to extract significant periodicities.
    
    Parameters
    ----------
    coef : pd.DataFrame
        DataFrame containing mean wavelet coefficients for different periods
    x_val : array_like, pd.Series, pd.DatetimeIndex or DatetimeArray
        Time or x-axis values corresponding to the coefficients
        If set to datetime: time of each signal data point
        If set to numeric: array that indicates the index of each signal value 
                           inside a period (example: DOY for yearly signals)
    pstr : str or list of str
        Period band identifiers to extract peaks from (column names in coef)
    filter_val : int, float or list of [int, float], optional
        x_val values to keep while filtering peaks. 
        If x_val is a datetime, units are given in filter_per.
        If None, no filtering is applied
    filter_per : str, optional (used if x_val is a datetime)
        Units of filter_val, corresponding to a method of the pandas.Series.dt
    per : str, optional (used if x_val is a datetime) 
        Time periodicity of data for conversion when working with dates.
        Must correspond to a method of the pandas.Series.dt. Default is year.
    sgn : pd.DataFrame, optional
        DataFrame with signs of wavelet coefficients for additional filtering
    thresh : int or float, optional
        Threshold value for peak detection
    
    Returns
    -------
    peaks : pd.DataFrame
        DataFrame containing the identified peaks with relevant information
    speak : dict
        Dictionary containing the deleted peaks with relevant information
    """

    def validate_and_convert_filter_val(filter_val):
        """
        Validate filter_val and convert it to a dictionary if necessary.
        
        Args:
            filter_val: The input filter value, which can be:
                - None
                - A single number (int or float)
                - A list/tuple of numbers
                - A numpy array of numbers
                - A pandas Series of numbers
                - A dictionary with numeric values
        
        Returns:
            A dictionary where:
                - Keys are strings (e.g., 'rep')
                - Values are lists of numbers
                
        Raises:
            ValueError: If filter_val contains non-numeric values
        """
        if filter_val is None:
            return None
        
        # Convert single numbers to list
        if isinstance(filter_val, (int, float)):
            filter_val = [filter_val]
        
        # Convert list/tuple to dictionary
        if isinstance(filter_val, (list, tuple)):
            if not all(isinstance(x, (int, float)) for x in filter_val):
                raise ValueError("filter_val must contain only integers or floats")
            return {'rep': list(filter_val)}
        
        # Convert numpy array to dictionary
        if isinstance(filter_val, np.ndarray):
            if filter_val.ndim != 1:
                raise ValueError("filter_val must be a 1D array")
            if not np.issubdtype(filter_val.dtype, np.number):
                raise ValueError("filter_val must contain only numbers")
            return {'rep': filter_val.tolist()}
        
        # Convert pandas Series to dictionary
        if isinstance(filter_val, pd.Series):
            if not pd.api.types.is_numeric_dtype(filter_val):
                raise ValueError("filter_val must contain only numbers")
            return {'rep': filter_val.tolist()}
        
        # Convert pandas DataFrame to dictionary using column names as keys
        if isinstance(filter_val, pd.DataFrame):
            filter_dict = {}
            for col in filter_val.columns:
                series = filter_val[col]
                if not pd.api.types.is_numeric_dtype(series):
                    raise ValueError(f"filter_val column '{col}' must contain only numbers")
                filter_dict[col] = series.tolist()
            return filter_dict
        
        # Validate dictionary
        if isinstance(filter_val, dict):
            for key, value in filter_val.items():
                if isinstance(value, (int, float)):
                    filter_val[key] = [value]
                elif isinstance(value, (list, tuple)):
                    if not all(isinstance(x, (int, float)) for x in value):
                        raise ValueError(f"filter_val[{key}] must contain only integers or floats")
                elif isinstance(value, np.ndarray):
                    if value.ndim != 1:
                        raise ValueError(f"filter_val[{key}] must be a 1D array")
                    if not np.issubdtype(value.dtype, np.number):
                        raise ValueError(f"filter_val[{key}] must contain only numbers")
                    filter_val[key] = value.tolist()
                elif isinstance(value, pd.Series):
                    if not pd.api.types.is_numeric_dtype(value):
                        raise ValueError(f"filter_val[{key}] must contain only numbers")
                    filter_val[key] = value.tolist()
                else:
                    raise ValueError(f"filter_val[{key}] must be numerical data")
            return filter_val
        
        raise ValueError('filter_val must be numerical data')

    def find_max(
        y: np.ndarray, 
        *x_args: Union[np.ndarray, pd.Series], 
        height: Optional[float] = None, 
        distance: Optional[int] = None, 
        threshold: Optional[float] = None
    ) -> Union[Tuple[np.ndarray], Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Find peaks in a signal and return their properties.
        
        This function is a wrapper around scipy.signal.find_peaks that allows for multiple
        x-coordinates to be returned corresponding to the peak locations in y.
        
        Parameters
        ----------
        y : np.ndarray
            1D array of signal values to find peaks in
        *x_args : Union[np.ndarray, pd.Series]
            Optional arrays of x-coordinates for each peak
        height : Optional[float], default=None
            Required height of peaks
        distance : Optional[int], default=None
            Required minimal horizontal distance between neighboring peaks
        threshold : Optional[float], default=None
            Required threshold of peaks
        
        Returns
        -------
        peaks_idx : np.ndarray
            Indices of peaks in the original signal
        peak_props : dict
            Dictionary with peak properties (only if height is specified)
        x_peaks : List[np.ndarray]
            List of arrays with x-coordinates for each peak (if x_args provided)
        """
        from scipy.signal import find_peaks
        
        # Find peaks in the signal
        if height is None and distance is None and threshold is None:
            peaks_idx = find_peaks(y)[0]
        else:
            # Create kwargs dictionary for parameters that are not None
            kwargs = {}
            if height is not None:
                kwargs['height'] = height
            if distance is not None:
                kwargs['distance'] = distance
            if threshold is not None:
                kwargs['threshold'] = threshold
                
            # Find peaks with specified parameters
            peaks_idx, props = find_peaks(y, **kwargs)
        
        # If no x-coordinates provided, return only peak indices
        if not x_args:
            if height is not None:
                return y[peaks_idx], props
            else:
                return y[peaks_idx]
        
        # Otherwise, extract x-coordinates at peak positions
        x_peaks = []
        for x in x_args:
            if isinstance(x, pd.Series):
                x = x.values
            x_peak = x[peaks_idx]
            x_peaks.append(x_peak)

        # return (yP, *xP_list)
        
        # Return peak indices and x-coordinates
        if height is not None:
            return y[peaks_idx], props, *x_peaks
        else:
            return y[peaks_idx], *x_peaks


    # Validate filter_val and convert to dict
    filter_val = validate_and_convert_filter_val(filter_val)

    #  Define periodicity
    if isinstance(x_val[0],(np.datetime64, datetime, pd.Timestamp)):
        # Convert to pandas Series for consistent .dt access
        x_val = pd.Series(x_val)

        # Valid methods of Series.dt
        valid_reps = ['year', 'month', 'day', 'hour', 'minute', 'second', 
                      'microsecond', 'nanosecond', 'dayofweek', 'dayofyear', 'quarter']
        
        # Check if units for periods admissible  
        if per not in valid_reps:
            raise ValueError(f"Invalid input periodic value {per}."
                            f"Must be one of {valid_reps}")

        # Define index of periods
        idx = getattr(x_val.dt, per)
        # nb_pers = np.sum(idx_change) + 1

        # Define filtering function
        if filter_val is not None:
            # Check if units for filtering values specified
            if filter_per is None:
                raise ValueError('filter_per must be specified, to indicate units of filter_val'
                                f'\nValues may be chosen among {valid_reps}')   
            # Check if units for filtering values admissible  
            if filter_per not in valid_reps:
                raise ValueError(f"Invalid input periodic value {filter_per}."
                                f"Must be one of {valid_reps}")
            # Define filtering function
            def filter_pers(coef_in, filt_id):
                # if filter_val is not None:
                period_values = getattr(coef_in['loc'].dt, filter_per)
                return coef_in[period_values.isin(filt_id)][['idx', 'loc', 'pk']]
    else:
        # Convert to numpy
        x_val = np.asarray(x_val)
        # Check if entire values given
        if not np.issubdtype(x_val.dtype, np.integer):# or not np.all(x_val >= 0):
            x_val2 = x_val
            x_val = x_val.astype(int)
            if np.sum(x_val2 - x_val) != 0:
                raise TypeError('x_val must contain only integer indices, representing the index of each signal value in a period'
                                '\nExample: DOY for yearly periods')
        
        # Define index of periods
        idx_change = np.diff(x_val) < 0
        # nb_pers = np.sum(idx_change) + 1
        idx = np.cumsum(idx_change)

        # Define filtering function
        if filter_val is not None:
            # Define filtering function
            def filter_pers(coef_in, filt_id):
                period_values = coef_in['loc']
                return coef_in[period_values.isin(filt_id)][['idx', 'loc', 'pk']]
        
    
    # Sign of wavelet coefficients (not used)
    if sgn is None:
        sgn = pd.DataFrame(np.nan, index=x_val, columns=[pstr])
    
    # Names of output tables
    column_names = [f"{identifier}{suffix}" for identifier in pstr for suffix in ("loc", "value")]
    
    # Initialize peaks dataframe
    unique_idx = np.unique(idx)

    # Ensure pstr is iterable
    if isinstance(pstr, str):
        pstr = [pstr]

    # Initialize all peaks dictionary
    all_peaks = {pid: pd.DataFrame() for pid in pstr}

    if filter_val is None:
        rep_peak = None
        speaks = None
    else:
        # Flag for warning
        warn1 = dict(zip(filter_val, [dict(zip(pstr, [[]for _ in range(len(pstr))])) for _ in range(len(filter_val))]))
        warn2 = dict(zip(filter_val, [dict(zip(pstr, [[]for _ in range(len(pstr))])) for _ in range(len(filter_val))]))

        # Initialize repeating peaks dictionary  with pre-allocated DataFrames
        rep_peak = {filtid: pd.DataFrame(index=unique_idx.astype(str), columns=column_names) for filtid in filter_val}

        # Initialize speaks dictionary with pre-allocated DataFrames
        speaks = {pid: pd.DataFrame() for pid in pstr}

    # Loop through all desired periods
    for j, pid in enumerate(pstr):
        # Find maxima in the coefficient series
        pk, loc, idx2, wave = find_max(coef[pid].values, x_val, idx, sgn[pid].values)
        if len(pk) == 0:
            continue
    
        # Create a dataframe of peaks
        coef_max = pd.DataFrame({'idx': idx2, 'loc': loc, 'pk': pk, 'wave': wave})
        
        # Threshold to delete peaks
        coef_max_thresh = coef_max[coef_max['pk'] > thresh]
        if coef_max_thresh.empty:
                continue
        
        # All peaks
        all_peaks[pid] = coef_max_thresh

        # Repeating peaks
        if filter_val is not None:
            # Initialize filtered peaks dictionary with pre-allocated lists
            # speak = {filtid: [] for filtid in filter_val}
            speak = []
            # Loop through all filter values
            for filt_id, filt_value in filter_val.items():
                # Repeating peaks
                idcoef = filter_pers(coef_max_thresh, filt_value)
                if idcoef.empty:
                    continue

                # Check for repeated values
                unique_values, counts = np.unique(idcoef['idx'], return_counts=True)
                
                # If more than one period per year
                if np.any(counts>1):  
                    # Define repeated values
                    repeated_values = unique_values[counts > 1]
                    warn_vals1 = []
                    warn_vals2 = []
                    for i in repeated_values:
                        # Try to filter using higher period if available
                        filt_idx = (idcoef['idx'] == i)
                        date_pks = idcoef.loc[filt_idx, 'loc']
                        # Find closest peak to the higher period peak
                        pid_prev = pstr[j-1]
                        prev_date = rep_peak[filt_id].loc[str(i), f"{pid_prev}loc"]
                        if j==0 or pd.isna(prev_date):
                            warn_vals2.append(i.astype(str))
                            # keep_idx = np.argmax(idcoef.loc[filt_idx, 'pk'])
                            keep_idx = idcoef.loc[filt_idx, 'pk'].idxmax()
                        else:
                            warn_vals1.append(i.astype(str))
                            keep_idx = np.abs(date_pks - prev_date).idxmin()
                            # keep_idx = np.argmin(np.abs(date_pks - prev_date))
                                
                        # Keep only the closest (or max) peak
                        to_remove = idcoef.index[filt_idx & (idcoef['loc'] != idcoef.loc[keep_idx, 'loc'])]
                        # to_remove = idcoef.index[filt_idx & (idcoef['loc'] != date_pks.iloc[keep_idx])]
                        idcoef = idcoef.drop(to_remove)
                        
                    # Warnings
                    if any(warn_vals1):
                        warn1[filt_id][pid] = warn_vals1
                    if any(warn_vals2):
                        warn2[filt_id][pid] = warn_vals2
                    
                # Store peaks
                rep_peak[filt_id].loc[idcoef.idx.map(str), [f"{pid}loc", f"{pid}value"]] = idcoef[['loc', 'pk']].values

                # Store deleted peaks for filt_id
                filtered_peaks = coef_max_thresh[~coef_max_thresh['loc'].isin(idcoef['loc'])]
                speak.append(filtered_peaks)
                # speak[filt_id].append(filtered_peaks)

            # Combine deleted peaks
            # Pre-compute the list of DataFrames for each pid
            # df_list = [df for df in speak.values() if not df.empty]
            
            # Store deleted peaks
            if speak:  # Only concatenate if there are non-empty DataFrames
                # Concatenate once with pre-computed list
                peaks_data = pd.concat(speak, ignore_index=True)
                
                if len(filter_val) > 1:
                    # Keeping only filtered peaks for each value in filter_val
                    counts = peaks_data.groupby(peaks_data.columns.tolist(), as_index=False).size()
                    repeated_peaks = peaks_data.merge(counts, on=peaks_data.columns.tolist())
                    speaks[pid] = repeated_peaks[repeated_peaks['size'] == len(filter_val)].drop('size', axis=1).reset_index(drop=True)

    # Warnings
    if filter_val is not None:
        for filt_id in filter_val:
            w1 = warn1[filt_id]
            w2 = warn2[filt_id]
            w1_exists = not all(len(values) == 0 for values in w1.values())
            w2_exists = not all(len(values) == 0 for values in w2.values())

            if w1_exists or w2_exists:
                import warnings
                warning_message = f"For {filt_id}: multiple peaks found inside periods.\n"
                
                if w1_exists:
                    warning_message += f"Peaks closest to previous (higher) period (e.g. '{pid_prev}' for '{pid}') retained for:\n"
                    for key, values in w1.items():
                        if values:
                            warning_message += f"{key}: {', '.join(map(str, values))}\n"
                
                if w2_exists:
                    warning_message += "Due to missing previous period peaks, highest peak value retained for:\n"
                    for key, values in w2.items():
                        if values:
                            warning_message += f"{key}: {', '.join(map(str, values))}\n"
                    if pstr[0] in w2.items():
                        warning_message += 'Please add higher periods to enable tracking peaks in lower periods.\n'
                
                warnings.warn(warning_message, UserWarning, stacklevel=3)

        # # Group all Peaks and filtered peaks
        # all_peaks = {}
        # speaks = {}
        # for dtb in pstr:
        #     # Concatenate peaks, count, delete and sort repeated peaks
        #     peaks_data = pd.concat(speak[pid][filt_id], ignore_index=True)
        #     counts = peaks_data.groupby(peaks_data.columns.tolist()).size()
        #     peaks_data = peaks_data.drop_duplicates().reset_index(drop=True)
        #     peaks_data = peaks_data.sort_values(by=['idx', 'ts']).reset_index(drop=True)
        #     # Store all peaks
        #     all_peaks[dtb] = peaks_data
        #     # Store deleted peaks
        #     speaks[dtb] = peaks_data[counts.values == len(speak.keys())].reset_index(drop=True)
        peak = {'all': all_peaks,
                'sup': speaks}
        peak.update(rep_peak)
    else:
        peak = {'all': all_peaks}

    return peak, rep_peak, speaks, all_peaks


def launch_wai(
    idx: Union[int, float, timedelta, ArrayLike, pd.Series, pd.DatetimeIndex, DatetimeArray],
    wavelet: Union[Dict[str, Any], str],
    per: Union[int, float, timedelta, str],
    ts: Union[int, float, timedelta],
    data: Union[ArrayLike, pd.Series, pd.DataFrame],
    xdata: Optional[Union[ArrayLike, pd.Series, pd.DatetimeIndex, DatetimeArray]] = None,
    str2val: Optional[Union[int, float, timedelta, Dict[str, Union[int, float]]]] = None,
    units: Optional[str] = None,
    peak: Optional[Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]] = None,
    perpeak: Optional[str] = None,
    filter_dates: Optional[Union[int, float, ArrayLike, pd.Series]] = None,
    filter_per: Optional[str] = None,
    plotting_mode: str = 'together',
    title: Optional[str] = 'default',
    xlims: Optional[Union[ArrayLike, pd.Series]] = None
) -> None:
    """
    Launch the visualization for Wavelet Area Interpretation (WAI).
    
    This function processes the input parameters and manages the plotting of wavelet
    coefficients for specified time and period combinations.
    
    Parameters
    ----------
    idx : int, float, timedelta, str, array_like, pd.Series, pd.DatetimeIndex, DatetimeArray
        Indices or positions to analyze, either as a value, timedelta, or string identifier.
        If (1) per is not a string, or (2) perpeak is not specified, or (3) no peak dataframe is provided, 
        idx indicates the indices (x-values) of the signal to be analyzed.
        Otherwise, idx must indicate idx values to be used from the chosen peak dataframe:
        For all/sup peaks, it must correpond to values in column 'idx'
        For repeated peaks, it must correspond to index values of the peak dataframe
    wavelet : dict or str
        Dictionary containing wavelet parameters and functions, or a string
        identifier for a pre-defined wavelet.
    per : int, float, timedelta or str
        Period to analyze, either as a value, timedelta, or string identifier.
        If str, str2val must indicate the value of the period, either by providing
        the value directly or by providing a dictionary with the mapping (output of coefvalues)
    ts : int, float or timedelta
        Sampling period.
    data : np.ndarray, pd.Series, pd.DataFrame
        The series data. If a DataFrame/Series is given, xdata (if missing) values are taken from the index
        or from columns 'xdata' or 'time'/'datetime' (if datetime).
    xdata : np.ndarray, pd.Series, pd.DatetimeIndex or DatetimeArray, optional (default=None)
        The x-axis values corresponding to the data. If None and no xdata extracted from data, 
        xdata is set to np.arange(len(data)).
    str2val : int, float, timedelta or dict, optional (default=None)
        Dictionary mapping period identifiers (given in 'per') to period values. 
        If numeric or timedelta, considered as the value mapped to 'per' string.
        Ignored if per is numeric or timedelta.
    units : str, optional (default=None)
        Time unit for period conversion, used if per is a string identifier
    peak : dict[str, pd.DataFrame], optional (default=None)
        Dictionary containing all peaks, filtered peaks and repeated peaks.
    perpeak : float or str, optional (default=None)
        Period to use for peak visualization, among field names in peaks ('all', 'sup' 
        and identifiers for repeated peaks, e.g. 'apr', 'jun' or 'nov').
    filter_per : str, optional (default=None)
        Period used for filtering.
        Ignored if xdata is not a datetime.
    filter_dates : list of int or float, optional (default=None)
        Value used for filtering peaks.
        Ignored if xdata is not a datetime.
    plotting_mode : str, optional (default='together')
        Plotting mode: 'together' or 'separate'
    title : str, optional (default='default')
        Title for the plot

    Returns
    -------
    None
        This function produces visualizations but does not return any values
    
    Notes
    -----
    - The function supports both numeric and datetime inputs for timestamps.
    - When mode='together', all provided indices are visualized in the same plot.
    - When mode='separate', each index is visualized in a separate plot.
    """

    def validate_and_convert_inputs(idx, data, xdata, xlims, per, str2val, units, ts, wavelet, plotting_mode, title):
        """
        Validate idx and convert it to a list/tuple.
        
        Args:
            idx: The input idx value, which can be:
                - None
                - A single number (int or float) or datetime
                - A list/tuple of numbers or datetime
                - A numpy array of numbers or datetime
                - A pandas Series of numbers or datetime
        
        Returns:
            A list/tuple of numbers
                
        Raises:
            ValueError: If filter_val contains non-numeric values
        """
        # 1. idx
        if idx is None:
            raise ValueError("idx must be numeric/datetime or a list of numeric/datetime values")
        # Convert single numbers to list
        if isinstance(idx, (int, float, datetime, np.datetime64, pd.Timestamp)):
            idx = [idx]
        # Convert numpy array or pandas Series to list
        elif isinstance(idx, (np.ndarray, pd.Series, pd.DatetimeIndex, DatetimeArray)):
            if idx.ndim != 1:
                raise ValueError("idx must be a 1D array")
            idx = idx.tolist()
        # Validate list/tuple
        if isinstance(idx, (list, tuple)):
            if not all(isinstance(x, (int, float, datetime, np.datetime64, pd.Timestamp)) for x in idx):
                raise ValueError("idx must contain only integers, floats or datetime")
        else:
            raise ValueError('idx must be numerical/datetime or a list of numerical/datetime values')

        # 2. data and xdata
        # Convert data to numpy array if it's not already + extract xdata from DataFrame
        if isinstance(data, pd.DataFrame):
            # Handle pandas DataFrame
            if data.empty:
                raise ValueError('data must contain at least one signal (DataFrame is empty)')
            cols = data.columns
            
            # Try to extract xdata information if missing
            if xdata is None:
                columns_lower = [col.lower() for col in cols]
                # Look for xdata column
                if 'xdata' in columns_lower:
                    xdata = data[cols[columns_lower.index('xdata')]]
                    print('Using xdata column found in DataFrame as xdata')
                # Look for time columns with standard names
                elif 'time' in columns_lower:
                    xdata = data[cols[columns_lower.index('time')]]
                    print('Using time column found in DataFrame as xdata')
                elif 'datetime' in columns_lower:
                    xdata = data[cols[columns_lower.index('datetime')]]
                    print('Using datetime column found in DataFrame as xdata')
                # If no time column, take index
                else:
                    xdata = data.index
                    print('Using DataFrame index as xdata')
            # Get signal data from DataFrame
            data = data.iloc[:, 0].to_numpy()
        elif isinstance(data, pd.Series):
            # Try to extract xdata information if missing
            if xdata is None:
                xdata = data.index
                print('Using Series index as xdata')
            # Get signal data from Series
            data = data.to_numpy()
        elif not isinstance(data, np.ndarray):
            # Convert to numpy array
            try:
                if all(isinstance(x, int) for x in data):
                    data = np.array(data, dtype=int)
                else:
                    data = np.array(data, dtype=float)
            except (ValueError, TypeError):
                raise TypeError("data must be convertible to a numeric numpy array")
        # Validate xdata
        if xdata is None:
            xdata = np.arange(len(data))
            print('Using numpy.arange(len(data)) as xdata')
        elif not isinstance(xdata, (np.ndarray, pd.Series, pd.DatetimeIndex, DatetimeArray)):
            # Convert to numpy array
            try:
                xdata = np.asarray(xdata)
            except (ValueError, TypeError):
                raise TypeError("xdata must be convertible to a numeric numpy array")
        elif xdata.ndim != 1:
            raise ValueError('xdata must be a 1D array')
        elif len(xdata) != len(data):
            raise ValueError('xdata must have the same length as data')

        # 3. per, str2val and units
        # Check and convert period if needed
        if isinstance(per, str):
            colname = per
            # Validate str2val dictionary and convert period
            if str2val is None:
                raise ValueError(f'Identifier {per} given but no dictionary (str2val) for correspondence')
            elif isinstance(str2val, (float, int, timedelta)):
                per = str2val  # Period given in str2val
            elif not isinstance(str2val, dict):
                raise TypeError('str2val must be either a numeric value representing the period in per '
                            f'or a dictionary giving the period in field {per}')
            elif per not in str2val.keys():
                raise ValueError(f'Identifier {per} must be chosen among keys of str2val. '
                                f'Available identifiers and corresponding periods: {str2val}')
            else:
                per = str2val[per]  # Period
        else:
            colname = None
        
        # 4. ts and units
        # Convert to adimensional time
        if isinstance(ts, timedelta):
            # Convert ts to adimensional time
            dt, _, dtformat = data_utils.extract_duration_info(ts)
            # Convert per to adimensional time (if conversion not provided already)
            if colname is None:  # Only if conversion not provided
                if not isinstance(per, timedelta):
                    raise TypeError(f'Period to study ({type(per).__name__}) must be the same type as ts ({type(ts).__name__})')
                # Create display string
                per_display = per
                # Convert per to adimensional time
                per = data_utils.convert_timdelta_array(dtformat, per)[0].item()
            else:
                # Validate units
                if units is None:
                    if 'units' not in str2val:
                        raise ValueError('For string input "per", units argument must be specified either '
                            'as input or in str2val dictionary for timedelta conversion')
                    units = str2val['units']
                if not isinstance(units, str):
                    raise TypeError('units must be a string')
                # Convert per to timedelta according to units
                per = data_utils.get_as_timedelta(units, per)
                # Create display string
                per_display = per
                # Convert per to adimensional time, using the same time unit as ts
                per = data_utils.convert_timdelta_array(dtformat, per)[0].item()
            
            if title == 'default':
                # Create display string for timedelta
                days = per_display.days
                hours, remainder = divmod(per_display.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                milliseconds = per_display.microseconds // 1000
                microseconds = per_display.microseconds

                # Determine the highest unit to display
                if days > 0:
                    per_display = f'{days} days'
                elif hours > 0:
                    per_display = f'{hours}.{seconds / 3600:.2f} hours'  # Include remainder in hours
                elif minutes > 0:
                    per_display = f'{minutes}.{seconds / 60:.2f} minutes'  # Include remainder in minutes
                elif seconds > 0:
                    per_display = f'{seconds} seconds'
                elif milliseconds > 0:
                    per_display = f'{milliseconds} milliseconds'
                else:
                    per_display = f'{microseconds} microseconds'
        else:
            if not isinstance(per, (int, float)) and not isinstance(ts, (int, float)):
                raise TypeError(f'Period to study ({type(per).__name__}) must be the same type as ts ({type(ts).__name__})')
            dt = ts
            per_display = f'{int(per) if per.is_integer() else per}'

        # 5. wavelet
        # Validate wavelet dictionary
        if isinstance(wavelet, str):
            import wt_utils
            wavelet = wt_utils.define_wavelet(wavelet)
        elif not isinstance(wavelet, dict):
            raise TypeError('wavelet must be a dictionary resulting from function '
                            'wt_utils.define_wavelet or a string identifier for the wavelet used')

        # 6. Validate string for plotting mode
        if plotting_mode not in ('together', 'separate'):
            raise ValueError(f'plotting_mode must be "together" or "separate", not {plotting_mode}')
        
        # 7. Validate title
        if title is not None and not isinstance(title, str):
            raise TypeError(f'title must be a string, not {type(title).__name__}')
        
        # 8. Validate xlims
        if xlims is not None:
            # Convert xlims to array
            if not isinstance(xlims,(tuple, list)) and not hasattr(xlims, '__array__'):
                raise ValueError('xlims must be a list, tuple or array of length 2')
            else:
                xlims = np.asarray(xlims).flatten()
            # Check that xlims is of length 2
            if len(xlims) != 2:
                raise ValueError('xlims must be a list, tuple or array of length 2')
            # Check that xlims and xdata are of the same type
            if isinstance(xdata[0],(datetime, np.datetime64, pd.Timestamp)): 
                if not isinstance(xlims[0],(datetime, np.datetime64, pd.Timestamp)):
                    raise TypeError('xlims must be of the same type as xdata')
            elif not isinstance(xdata[0],(int, float)) and not isinstance(xlims[0],(int, float)):
                if xdata.dtype != xlims.dtype:
                    raise TypeError('xlims must be of the same type as xdata')  
            # Check that xlims values are ascending
            if xlims[0] > xlims[1]:
                raise ValueError('xlims[0] must be less than '
                               'xlims[1] for zooming to be valid')
            # Check that xlims are in the range of xdata
            if xlims[0] < xdata[0] or xlims[1] > xdata[-1]:
                raise ValueError('xlims must be in the range of xdata')

        return idx, data, xdata, xlims, per, per_display, colname, dt, wavelet

    # Validate inputs
    idx, data, xdata, xlims, per, per_display, colname, dt, wavelet = validate_and_convert_inputs(
        idx, data, xdata, xlims, per, str2val, units, ts, wavelet, plotting_mode, title)

    # Define scale to study
    a = per / wavelet['fourier_fac'] # Scale
        
    # Define x-position to study
    if perpeak is not None:
        # Convert idx to appropriate format if it's a range of years
        

        # if peak is None:
        #     raise ValueError('Missing peak dictionary but perpeak specified')
        # elif not isinstance(peak, dict):
        #     raise TypeError(f'peak must be a dict, not a {type(peak).__name__}')
        # elif not 'colname' in locals():
        #     raise NameError('per must indicate colnames is not defined. Please specify column names for the output')
        #
        #  Get idxs from peak
        # if perpeak in peak.keys():
        #     if perpeak in ['sup','all']: # filtered or all peaks
        #         peak_locs = peak[perpeak][per]
        #         peak_locs = peak_locs[[v in str(idx) for v in peak_locs['idx']]]
        #         peak_index = peak_locs['idx']
        #     else:  # repeated peaks
        #         peak_locs = peak[perpeak][f"{colname}loc"].dropna()
        #         peak_locs = peak_locs[[v in str(idx) for v in peak_locs.index]]
        #         peak_index = peak_locs.index
        # else:
        #     raise ValueError(f'No key matching {perpeak} in peak dictionnary')

        if peak is None:
            raise ValueError('Missing peak dictionary but perpeak specified')
        elif not isinstance(peak, dict):
            raise TypeError(f'peak must be a dict, not a {type(peak).__name__}')
        elif colname is None:
            raise NameError('perpeak specified, but no string identifier in per indicating period to analyze.')

        # Get idxs from peak
        if perpeak not in peak.keys():
            raise ValueError(f'No key matching {perpeak} in peak dictionnary')
        elif perpeak in ['sup','all']:  # filtered or all peaks
            peak_locs = peak[perpeak][colname]['idx','loc']
            peak_locs = peak_locs[[v in idx for v in peak_locs['idx']]]
            peak_locs.set_index('idx', inplace=True)
        else:  # repeated peaks
            peak_locs = peak[perpeak][f"{colname}loc"].dropna()
            peak_locs = peak_locs[[v in str(idx) for v in peak_locs.index]]
            # peak_index = peak_locs.index.astype(int)

        if peak_locs.empty:
            raise ValueError(f"No '{perpeak}' peaks found for period '{per}'")

        # Filter peaks
        if filter_dates is not None:
            if not isinstance(filter_dates,(pd.Series, np.ndarray, list, tuple, int, float)):
                raise TypeError('filter_val must be numerical data')
        
            if isinstance(xdata[0],(datetime, np.datetime64, pd.Timestamp)):
                # if x_val is not None:
                #     if not isinstance(x_val[0],(datetime, np.datetime64, pd.Timestamp)) or sum(x_val != xdata) > 0:
                #         print('Specifying x_val is optional for datetime x values.'
                #             'If specified, x_val must be identical to xdata'
                #             'Ignoring x_val. xdata used instead for filtering')
                # Valid methods of Series.dt
                valid_reps = ['year', 'month', 'day', 'hour', 'minute', 'second', 
                        'microsecond', 'nanosecond', 'dayofweek', 'dayofyear', 'quarter']
                # Check if units for filtering values specified
                if filter_per is None:
                    raise ValueError('filter_per must be specified, to indicate units of filter_val'
                                    f'\nValues may be chosen among {valid_reps}')   
                # Check if units for filtering values and/or periods admissible  
                if filter_per not in valid_reps:
                    raise ValueError(f"Invalid input periodic value {filter_per}."
                                    f"Must be one of {valid_reps}")
                # Apply filtering
                period_values = getattr(pd.to_datetime(peak_locs).dt, filter_per)
                peak_locs = peak_locs[period_values.isin(filter_dates)]
                # Check if any peaks remaining after filtering
                if peak_locs.empty:
                    raise ValueError('After filtering, no peaks remaining for {per}')
            else:
            #     # Convert to numpy
            #     x_val = np.asarray(x_val)
            #     # Check if entire values given
            #     if not np.issubdtype(x_val.dtype, np.integer):
            #         x_val2 = x_val
            #         x_val = x_val.astype(int)
            #         if np.sum(x_val2-x_val) != 0:
            #             raise TypeError('x_val must contain only integer indices, representing the index of each signal value in a period'
            #                             '\nExample: DOY for yearly periods')
            #     if not np.all(x_val >= 0):
            #         raise ValueError('x_val must contain only positive indices, representing the index of each signal value in a period')
            #     # Define filtering function
            #     period_values = peak_locs.index
            #     peak_locs = peak_locs[period_values.isin(filter_val)]
                import warnings
                warnings.warn('filter_val is ignored since xdata is not a datetime.' 
                              'Variable idx may be used for filtering', UserWarning, stacklevel=2)
    else:
        peak_locs = pd.Series(idx)
    
    # Zooming
    if xlims is not None:
        # Keep only data within xlims
        data = data[(xdata >= xlims[0]) & (xdata <= xlims[1])]
        xdata = xdata[(xdata >= xlims[0]) & (xdata <= xlims[1])]
        # Delete peaks outside xlims
        peak_locs = peak_locs[(peak_locs >= xlims[0]) & (peak_locs <= xlims[1])]
        # Check if any peaks remaining after filtering
        if peak_locs.empty:
            raise ValueError(f'After zooming on [{xlims}], no peaks remaining for {per}')
    
    # Create adimensional time vector
    t = np.arange(len(data))
    
    # Process each idx
    if plotting_mode == 'together' and len(peak_locs) > 1 and peak_locs.index.is_unique: # Plot all idxs together
        # Initialize
        Tb = None
        for b in peak_locs:
            # Initialize output
            temp = compute_wave(data, xdata, b, a, dt, wavelet, t)
            if Tb is None:
                Tb = temp
            else:
                diff = np.abs(temp['wav']) - np.abs(Tb['wav'])
                j = np.where(diff > 1e-5)[0][0]
                Tb.iloc[j:] = temp.iloc[j:]
        # Determine title if default
        if title == 'default':
            title = f"WAI ({wavelet['name']}) - Period: " + per_display
        plot_wai(Tb, wavelet, title)
    else:
        # Plot each idx separately
        
        if title == 'default':
            for b in peak_locs:
                plot_title = f"WAI ({wavelet['name']}) - Period: " + per_display + f", Index: {b}"
                
                Tb = compute_wave(data, xdata, b, a, dt, wavelet, t)
                plot_wai(Tb, wavelet, plot_title)
        else:
            for b in peak_locs:
                Tb = compute_wave(data, xdata, b, a, dt, wavelet, t)
                plot_wai(Tb, wavelet, title)
    
    return None

def compute_wave(
    data: np.ndarray,
    xdata: Union[np.ndarray, pd.Series, pd.DatetimeIndex, DatetimeArray],
    b: Union[int, float, datetime, np.datetime64, pd.Timestamp],
    a: Union[int, float],
    dt: Union[int, float],
    wavelet: Dict[str, Any],
    t: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compute wavelet transform at a specific time and scale.
    
    This function performs the wavelet transform calculation for a signal at a specific
    point in time (b) and scale (a), extracting the localized signal characteristics.
    
    Parameters
    ----------
    data : np.ndarray
        The time series data
    xdata : np.ndarray, pd.Series, pd.DatetimeIndex or pd.arrays.DatetimeArray
        The x-axis values (time) corresponding to the data
    b : int, float, datetime, np.datetime64, or pd.Timestamp
        The time point to analyze
    a : int or float
        The scale parameter of the wavelet
    dt : int or float
        The sampling period in normalized units
    wavelet : dict
        Dictionary containing wavelet parameters and functions
    t : np.ndarray, optional (default=None)
        The time vector (if None, will be computed from data length)
    
    Returns
    -------
    result : pd.DataFrame
        DataFrame containing the computed wavelet transform and related parameters:
        - 'xdata': x-axis values
        - 'data': original data
        - 'wav': wavelet function values
        - 'prod': product of wavelet and data
        - 'prod_pos_re': positive area (real part)
        - 'prod_neg_re': negative area (real part)
        - 'prod_pos_im': positive area (imaginary part, if applicable)
        - 'prod_neg_im': negative area (imaginary part, if applicable)
    
    Notes
    -----
    - If b is a datetime, it is converted to an index position.
    - The wavelet calculation uses the wavelet function (psi) from the wavelet dictionary.
    - The function handles both numeric and datetime inputs for the time point (b).
    """
    # Initialize output
    result = pd.DataFrame({'xdata': xdata, 'data': data})
    
    # Create time vector if not provided
    if t is None:
        t = np.arange(len(data))

    # Convert datetime b to index if needed
    if isinstance(b, (datetime, np.datetime64, pd.Timestamp)):
        if not isinstance(xdata[0], (datetime, np.datetime64, pd.Timestamp)):
            raise TypeError("If b is a datetime, xdata must also contain datetime values")
        
    # Find closest index to the specified time
    b_numeric = np.argmin(np.abs(xdata - b))  # Time position

    # Compute the wavelet function at the specified time and scale
    result['wav'] = wavelet['psi'](dt * (t - b_numeric) / a)
    
    # Compute product curves and areas
    result['prod'] = result['wav'] * result['data']  # Product curve
    result['prod_pos_re'] = np.maximum(np.real(result['prod']), 0)  # Positive area (real)
    result['prod_neg_re'] = np.minimum(np.real(result['prod']), 0)  # Negative area (real)
    if wavelet['is_complex']:
        result['prod_pos_im'] = np.maximum(np.imag(result['prod']), 0)  # Positive area (imag)
        result['prod_neg_im'] = np.minimum(np.imag(result['prod']), 0)  # Negative area (imag)
    return result

def plot_wai(
    result: pd.DataFrame,
    wavelet: Union[str, Dict[str, Any]],
    title: Optional[str] = None,
    fmt: Optional[str] = None
) -> None:
    """
    Plot the Wavelet Area Interpretation (WAI) visualization.
    
    This function creates a detailed visualization of the wavelet transform results,
    showing the time series data, wavelet function, and their interactions.
    
    Parameters
    ----------
    result : pd.DataFrame
        DataFrame containing the wavelet transform results from compute_wave()
    wavelet : dict
        Dictionary containing wavelet parameters and functions
    title : str, optional (default=None)
        Title for the plot
    fmt : str, optional (default=None)
        Format for x-axis, if datetime. Must be a valid matplotlib date format string.
    
    Returns
    -------
    None
        This function produces a visualization but does not return any values
    
    Notes
    -----
    - The function creates a 2- or 3-panel figure with subplots for different aspects of the transform
    - The top panel shows the original time series and the wavelet function
    - The second panel displays the product of the time series and (the real part of the) wavelet, highlighting regions where the wavelet and signal correlate
    - For complex wavelets, the third panel displays the product of the time series and the imaginary part of the wavelet.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import gridspec


    def set_wav_and_colors(wname):    
        if wname == 'mor':
            c = [
                [0, 0.45, 0.74],
                [0.07, 0.62, 1],
                [0.05, 0.24, 0.36],
                [0.85, 0.33, 0.1],
                [1, 0.41, 0.16],
                [0.44, 0.05, 0.12]
            ]
        elif wname == 'gau':
            c = [
                [0.64, 0.08, 0.18],
                [0.64, 0.08, 0.18],
                [0.64, 0.08, 0.18]
            ]
        elif wname.startswith('dog'):
            order = int(wname[3:])
            if order == 1:
                c = [
                    [0.16, 0.16, 0.61],
                    [0.07, 0.64, 0.95],
                    [0.05, 0.05, 0.47]
                ]
            else:
                c = [
                    [0.47, 0.67, 0.19],
                    [0.39, 0.83, 0.07],
                    [0.47, 0.67, 0.19]
                ]
        elif wname == 'haa':
            c = [
                [0.49, 0.18, 0.56],
                [0.72, 0.27, 1.00],
                [0.28, 0.12, 0.32]
            ]
    
        return c
    
    # Set colors
    c = set_wav_and_colors(wavelet['name'])
    no_plot = 2
    is_complex = wavelet['is_complex']
    if is_complex:
        no_plot += 1
    
    # Check if data is in datetime format
    change_format = False
    if fmt is not None and isinstance(result['xdata'][0], (datetime, np.datetime64, pd.Timestamp)):
        if not mdates.DateFormatter.validate(fmt):
            raise ValueError(f"Invalid date format string: {fmt}")
        change_format = True

    # Create figure and GridSpec for custom layout
    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(no_plot, 1, height_ratios=[2] + [1] * (no_plot - 1), hspace=0.3)
    # fig, axes = plt.subplots(no_plot, 1, figsize=(10, 6))
    
    # Plot signal and wavelet
    # Panel 1: Original signal and wavelet
    ax1 = plt.subplot(gs[0])
    # ax1 = axes[0]

    # Plot original signal
    ax1.plot(result['xdata'], result['data'], color=[0.9, 0.9, 0.9], linestyle='-', label='Signal')
    #     ax1.set_ylabel('Signal')
    if change_format:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
        plt.xticks(rotation=45)

    # Create twin axis for wavelet
    ax1b = ax1.twinx()

    # Plot wavelet
    if is_complex:
        # Create twin axis for wavelet
        ax1b.plot(result['xdata'], np.real(result['wav']), color=c[0], linestyle='-', linewidth=2, label=f'{wavelet["name"]} Wavelet (Real)')
        ax1b.plot(result['xdata'], np.imag(result ['wav']), color=c[3], linestyle='--', linewidth=2, label=f'{wavelet["name"]} Wavelet (Imag)')
    else:
        ax1b.plot(result['xdata'], result['wav'], color=c[0], linestyle='-', linewidth=2, label=f'{wavelet["name"]} Wavelet')
    # ax1.set_ylabel('Wavelet')
    # Add grid and legend
    ax1.grid(True, alpha=0.3)
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax1b.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Set y-labels and title
    ax1.set_ylabel('Signal', fontsize=10)
    ax1b.set_ylabel('Wavelet', fontsize=10)
    if title:
        ax1.set_title(title, fontsize=12)
    
    # Panel 2: Product of signal and wavelet
    ax2 = plt.subplot(gs[1], sharex=ax1)
    # ax2 = axes[1]
    
    # Plot areas
    ax2.fill_between(result['xdata'], result['prod_pos_re'], color=c[1], edgecolor=c[1], alpha=0.5, label='Positive Area')
    ax2.fill_between(result['xdata'], result['prod_neg_re'], color=c[2], edgecolor=c[2], alpha=0.5, label='Negative Area')
    # Plot areas
    ax2.fill_between(result['xdata'], result['prod_pos_re'], color=c[1], edgecolor=c[1], alpha=0.5, label='Positive Correlation')
    ax2.fill_between(result['xdata'], result['prod_neg_re'], color=c[2], edgecolor=c[2], alpha=0.5, label='Negative Correlation')

    # Add grid and legend
    ax2.grid(True, alpha=0.3)
    # ax2.legend(loc='upper right') 
    #ax2.legend([lgd[0], lgd[1]])

    # Set labels and format + plot imaginary area if wavelet is complex
    if is_complex:
        ax2.set_ylabel('Product curve (real part)', fontsize=10)
        # Panel 3: Areas
        ax3 = plt.subplot(gs[2], sharex=ax1)
        # ax3 = axes[2]
        ax3.fill_between(result['xdata'], result['prod_pos_im'], color=c[4], edgecolor=c[4], alpha=0.5, label='Positive Area')
        ax3.fill_between(result['xdata'], result['prod_neg_im'], color=c[5], edgecolor=c[5], alpha=0.5, label='Negative Area')
        # Add grid and legend
        ax3.grid(True, alpha=0.3)
        # ax3.legend(loc='upper right')
        ax3.set_ylabel('Product curve (imaginary part)', fontsize=10)
        # Set x-axis format
        if change_format:
            ax3.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
            plt.xticks(rotation=45)
    else:
        ax2.set_ylabel('Product curve', fontsize=10)
        # Set x-axis format
        if change_format:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
            plt.xticks(rotation=45)
    
    # Adjust layout manually
    # plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.3)

    # Display the figure
    plt.show()


def idxs_from_center_to_extremum_or_zero_dog(
    peak_in: Union[int, float, ArrayLike, pd.Series, pd.DataFrame],
    per: Union[str, int, float, timedelta],
    ts: Union[int, float, timedelta],
    wavelet: Union[Dict[str, Any],str],
    str2val: Optional[Union[Dict[str, float], float, int, timedelta]] = None,
    units: Optional[str] = None
) -> Union[int, float, np.ndarray, pd.Series, pd.DataFrame]:
    """
    Find indices of extrema or zero crossings from wavelet center indices.
    
    This function calculates the offset from wavelet centers to extrema or zero crossings
    based on the wavelet type and period.
    
    Parameters
    ----------
    peak_in : int, float, ArrayLike, pd.Series, pd.DataFrame
        Data containing locations of wavelet coefficients to be translated
    per : str, int, float timedelta
        Period to analyze, either as a string identifier, numeric value, or timedelta
    ts : int, float or timedelta
        Sampling period
    wavelet : dict or str
        Dictionary containing wavelet parameters and functions, or a string
        identifier for a pre-defined wavelet.
    str2val : dict, float, int or timedelta, optional (default=None)
        Dictionary mapping period identifiers (given in 'per') to period values. 
        If numeric or timedelta, considered as the value mapped to 'per' string.
        Ignored if per is numeric or timedelta.
    units : str, optional (default=None)
        Time unit for period conversion, used if per is a string identifier
    
    Returns
    -------
    peak_out : int, float, np.ndarray, pd.Series or pd.DataFrame
        Peak locations adjusted to extrema or zero crossings, in the same format as peak_in

    Raises
    ------
    ValueError
        If identifier is given but no dictionary for correspondence
        If identifier is not in dictionary keys
    TypeError
        If period and ts have incompatible types
        If peak_in has an invalid type
    KeyError
        If required column is not found in peak_in DataFrame

    Notes
    -----
    - For DOG wavelets, this function finds the offset from peak centers to extrema.
    - The function computes the appropriate translation based on wavelet properties.
    - The output maintains the same format as the input but with adjusted locations.
    """
    # Check and convert period to scale
    if isinstance(per, str):
        colname = str(per)
        if str2val is None:
            raise ValueError(f'Identifier {per} given but no dictionary (str2val) for correspondence.')
        if isinstance(str2val, (float, int, timedelta)):
            per = str2val  # Period given in str2val
        elif not isinstance(str2val, dict):
            raise TypeError('str2val must be either a numeric value representing the period in per '
                            f'or a dictionary giving the period in field {per}')
        elif per not in str2val.keys():
            raise ValueError(f'Identifier {per} must be chosen among keys of str2val. '
                            f'Available identifiers and corresponding periods: {str2val}')
        else:
            per = str2val[per]  # Period
        # Units conversion
        if isinstance(ts, timedelta):
            _, _, tsformat = data_utils.extract_duration_info(ts)
            # Validate units
            if units is None:
                if 'units' not in str2val:
                    raise ValueError('For string input "per", units argument must be specified either '
                        'as input or in str2val dictionary for timedelta conversion')
                units = str2val['units']
            if not isinstance(units, str):
                raise TypeError('units must be a string')
            per = data_utils.get_as_timedelta(units, per)
            # per = data_utils.convert_timdelta_array(tsformat, per)[0].item()
    else:
        colname = None

    # Matching types (per and ts)
    if type(per) != type(ts):
        if not isinstance(per, (int, float)) and not isinstance(ts, (int, float)):
            raise TypeError(f'Period to study must be the same type as ts ({type(ts).__name__})')
    
    # Validate wavelet dictionary
    if isinstance(wavelet, str):
        import wt_utils
        wavelet = wt_utils.define_wavelet(wavelet)
    elif not isinstance(wavelet, dict):
        raise TypeError('wavelet must be a dictionary resulting from function '
                        'wt_utils.define_wavelet or a string identifier for the wavelet used')
    # Check if wavelet is DOG
    if wavelet['name'][:3] != 'dog':
        import warnings
        warnings.warn('Translation from wavelet center to extremum or zero only applies to dog wavelets. '
                      'For other wavelet, this function applies a simple translation given by period speicified', UserWarning, stacklevel=2)
    
    # Extract desired peaks (standardize to a DataFrame)
    # Process input based on type
    if isinstance(peak_in, pd.DataFrame):
        if peak_in.shape[1] > 1:
            # Check if there is exactly one column containing "loc" in its name
            loc_columns = [col for col in peak_in.columns if "loc" in col]
            if len(loc_columns) == 1:
                # Set peak_out to the column and raise a warning
                peak_out = peak_in[loc_columns[0]]
                import warnings
                warnings.warn(f'Column "{loc_columns[0]}" was automatically chosen as it contains "loc" in its name.', UserWarning)
            elif colname is None:
                raise NameError('Column name to use for peak locations in peak_in not defined. '
                                'Please enter a Series with the column to use or specify the column name in per (and corresponding value in str2val)')
            elif f"{colname}loc" not in peak_in.columns:
                raise KeyError(f'Cannot find "{colname}loc" in columns of peak_in')
            peak_out = peak_in[f"{colname}loc"]
        else:
            peak_out = peak_in
    elif isinstance(peak_in, (tuple, list, np.ndarray, pd.Series)):
        peak_out = pd.Series(peak_in, name=f"{colname}loc")
    elif isinstance(peak_in, (float, int)):
        peak_out = pd.Series([peak_in], name=f"{colname}loc")
    else:
        # Convert to numpy array
        try:
            if all(isinstance(x, int) for x in peak_in):
                peak_out = pd.Series(np.array(peak_in, dtype=int), name=f"{colname}loc")
            else:
                peak_out = pd.Series(np.array(peak_in, dtype=float), name=f"{colname}loc")
        except (ValueError, TypeError):
            raise TypeError(f"peak_in must be convertible to a numeric numpy array. Got {type(peak_in).__name__}.")
    
    # Type checking for peak locations (matching ts)
    if isinstance(ts, timedelta):
        # Check if all non-NaN values in peak_out are datetime-like
        if not all(isinstance(val, (datetime, np.datetime64, pd.Timestamp)) for val in peak_out.dropna()): 
            raise TypeError(f'Location of peaks should follow ts format ({type(ts).__name__}). Must be datetime')
    elif type(peak_out.iloc[0]) != type(ts):
        if not isinstance(peak_out.iloc[0], (int, float)) and not isinstance(ts, (int, float)):
            raise TypeError(f'Location of peaks should follow ts format ({type(ts).__name__}).')
    
    # Translation
    b = per / wavelet['fourier_fac']  # Translation in units of per
    b_rem = b % ts  # Extract non-integer multiples of sampling period
    if b_rem < ts / 2:
        addvalue = b - b_rem  # Round down
    else:
        addvalue = b - b_rem + ts  # Round up
    
    # Move peak dates
    peak_out = peak_out + addvalue

    # Convert back to original format
    if isinstance(peak_in, pd.DataFrame):
        if peak_in.shape[1] > 1:
            return pd.DataFrame(peak_out)
        return peak_out
    elif isinstance(peak_in, (tuple, list, np.ndarray)):
        return peak_out.values
    elif isinstance(peak_in, pd.Series):
        return peak_out
    return peak_out.iloc[0]


def compute_support_period(
    wavelet: Union[Dict[str, Any],str],
    per: Union[int, float, timedelta],
    ts: Optional[timedelta] = None,
    percentout: float = 0.01,
    metric: str = 'int'
) -> Union[int, float, timedelta]:
    """
    Compute the support period of a wavelet for a given period.
    
    This function calculates the effective support of a wavelet at a specific scale,
    with adjustments for different wavelet types and formats.
    
    Parameters
    ----------
    wavelet : dict or str
        Dictionary containing wavelet parameters and functions, or a string
        identifier for a pre-defined wavelet.
    per : int, float or timedelta
        Period to compute support for, either as a numeric value or timedelta
    ts : timedelta, optional (default=None)
        Sampling period, required if per is a timedelta
    percentout : float, optional (default=0.1)
        Percentage of energy allowed outside the support
    metric : str, optional (default='int')
        Metric for computing support ('int' for integer support)
    
    Returns
    -------
    support : int, float or timedelta
        The computed support period in the same units as the input period
    
    Raises
    ------
    ValueError
        If ts is not specified when per is a timedelta
    TypeError
        If ts is not the same type as per
        If per is not a numeric value or timedelta
    
    Notes
    -----
    - For timedelta inputs, the function converts to adimensional values for computation.
    - The result is returned in the same format as the input period.
    - The wavelet's support function must be defined in the wavelet dictionary.
    """
    # Check and handle period format
    if isinstance(per, timedelta):
        if ts is None:
            raise ValueError('Need to specify sampling period (ts) for unit conversion')
        elif not isinstance(ts, timedelta):
            raise TypeError(f'ts ({type(ts).__name__}) must be the same type as per ({type(per).__name__})')
        # Check and convert ts and period to adimensionnal
        _ , fct, tsformat = data_utils.extract_duration_info(ts)
        _, _, performat = data_utils.extract_duration_info(per)
        per = data_utils.convert_timdelta_array(tsformat,per)[0].item()
    # Validate wavelet dictionary
    if isinstance(wavelet, str):
        import wt_utils
        wavelet = wt_utils.define_wavelet(wavelet)
    elif not isinstance(wavelet, dict):
        raise TypeError('wavelet must be a dictionary resulting from function '
                        'wt_utils.define_wavelet or a string identifier for the wavelet used')
    # Convert period
    if not isinstance(per,(int,float)):
        raise TypeError(f'per must be numerical, not a ({type(per).__name__})')
    
    # Compute scale
    s = per / wavelet['fourier_fac']
    # Compute support
    support = wavelet['support'](s, percentout, metric)
    if isinstance(ts, timedelta):
        # Convert back to original format
        return data_utils.convert_timdelta_array(performat,fct(support))[0].item()
    return support