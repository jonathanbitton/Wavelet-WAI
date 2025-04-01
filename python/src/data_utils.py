"""
Created on Fri Feb 14 2025

@author: Jonathan Bitton
"""
# Standard library imports
from typing import Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING
from datetime import datetime, timedelta

# Third-party imports
import numpy as np
from numpy.typing import ArrayLike

# Variable type hints
if TYPE_CHECKING:
    import pandas as pd
    from pandas.arrays import DatetimeArray

def get_args(**kwargs) -> Tuple[Dict[str, Union[str, int, float, timedelta]], 
                                Dict[str, Union[str, int, float]], 
                                Optional[Union[Dict[str, Union[int, float]],np.ndarray]], 
                                Dict[str, bool], 
                                Optional[np.ndarray]]:
    """
    Process and validate wavelet transform parameters.
    
    This function validates user inputs, handles defaults, resolves conflicts,
    and organizes parameters into logically grouped dictionaries for use in wavelet
    transform operations.
    
    Parameters
    ----------
    **kwargs : dict
        Keyword arguments for configuring the wavelet transform. Supported parameters include:
        (See Wavelet_Transform class docstring for detailed parameter descriptions)
        
        - Signal-related parameters:

        ts : float, int, timedelta
            Sampling period (for real time data) or x-axis difference between consecutive samples.
        fs : float, int
            Sampling frequency (alternative to ts).
        xdata : array_like
            Custom x-axis data.

        - Wavelet-specific parameters:

        wname : str
            Wavelet name ('mor', 'mex', 'dog{m}', 'gau', 'haa') or 
            wavelet set ('mor{w0}-{s2}', 'mex{s2}', 'dog{m}-{s2}', 'gau{s2}')
        norm : str
            Normalization method ('L1', 'L2').
        s2 : float, int
            Variance of the (enveloping) Gaussian function.
        w0 : float, int
            Central frequency for Morlet wavelet.

        - Scale-related parameters:

        dj : float, int
            Scale resolution (smaller = finer resolution).
        permin/pmin : float, int
            Minimum period to analyze.
        fmin : float, int
            Minimum frequency to analyze (alternative to permax).
        smin/s0 : float, int
            Minimum scale to analyze.
        permax/pmax : float, int
            Maximum period to analyze.
        fmax : float, int
            Maximum frequency to analyze (alternative to permin).
        smax : float, int
            Maximum scale to analyze.
        scaletype : str
            Scale spacing type ('lin', 'log', 'bot' for both).
        pcut : float, int
            Period cutoff for logarithmic to linear transition.
        fcut : float, int
            Frequency cutoff for transition (alternative to pcut).
        scut : float, int
            Scale cutoff for transition (alternative to pcut/fcut).
        dj_lin : float, int
            Scale resolution for linear scales portion.
        scales : array_like
            Explicit scale array (overrides other scale parameters).
            
        - Padding parameters:

        pad : str
            Padding method ('zpd', 'ref', 'sym', 'per', 'non').
        padmode : str
            Padding mode ('r', 'b', 'l' for right, both, left).

        - COI-related parameters:

        coimethod : str
            COI calculation method ('ana', 'num' for analytic and numerical).
        coimetric : str
            COI metric ('int', 'ene' for integration and energy).
        percentout : float
            Threshold to define COI.
            
        - Flags:

        figure : bool
            Whether to generate figures.
        signif : bool
            Whether to calculate significance.
        coi : bool
            Whether to calculate cone of influence.
        use_pyfftw : bool
            Whether to use pyfftw for FFT.
        parallel_proc : bool
            Whether to use parallel processing for computing wavelet coefficients.
        
        - Parallelization parameters:

        workers : int
            Number of workers for parallel processing.
        chunk_size : int
            Chunk size for parallel processing.

        - Display parameters:

        plotunits : str
            Time units for plotting
        
        - Calculated parameters:
        dt : float, calculated
            Adimensionnal time step or x-axis difference between consecutive samples (derived from ts or fs).
        dt_fct : float, calculated
            Function to convert numerical values back to timedelta.
    
    Returns
    -------
    params : dict
        General parameters including time specs, padding, etc.
    wav_params : dict
        Wavelet-specific parameters (wname, norm, s2, w0).
    scale_param : dict or numpy.ndarray
        Temporary scale parameters (permin, smin, permax, smax, pcut) or scales array.
    flags : dict
        Boolean flags (coi, signif, figure).
    processing : dict
        Processing parameters (use_pyfftw, parallel_proc, workers, chunk_size).
    xdata : array_like or None
        X-axis data if provided.
        
    Raises
    ------
    ValueError
        If invalid parameters, invalid parameter combinations, calculated parameters, or conflicting parameters are provided.
    TypeError
        If parameters are of incorrect type.
        
    Notes
    -----
    - Handles parameter aliases (e.g., s0/smin, pmin/permin) for backward compatibility
    - Resolves conflicts between exclusive parameters
    - Applies appropriate defaults based on context
    - Calculates derived parameters (e.g., permin from fmax) when needed
    - Validates parameter combinations for consistency
    - Handles time unit conversions for timedelta objects
    """
    import warnings

    # Define valid options for constrained parameters
    VALID_OPTIONS = {
        'wname': {'mor', 'mex', 'dog', 'gau', 'haa'},
        'scaletype': {'lin', 'log', 'bot'},
        'norm': {'L1', 'L2'},
        'pad': {'zpd', 'ref', 'sym', 'per', 'non'},
        'padmode': {'r', 'b', 'l'},
        'coimethod': {'ana', 'num'},
        'coimetric': {'int', 'ene'},
        'plotunits': {'microsec', 'millisec', 'sec', 'min', 'hour', 'day'}
    }

    # Parameter definitions with validation rules and default values
    param_specs = {
        # Basic parameters
        'ts': {'default': None, 'type': (float, int, timedelta)},
        'fs': {'default': None, 'type': (float, int)},
        'dt': {'default': None, 'calculated': True, 'error_msg': 'dt is calculated internally, please specify argument ts instead'},
        'dt_fct': {'default': None, 'calculated': True, 'error_msg': 'dt_fct is calculated internally, if ts is a timedelta'},
        
        # Scale parameter control
        'dj': {'default': 1/20, 'type': (float, int), 
            'validate': lambda v: v > 0,
            'error_msg': 'Invalid dj value "{value}". Please specify a positive value'},
        'permin': {'default': None, 'type': (float, int), 'alias': 'pmin'},
        'fmin': {'default': None, 'type': (float, int)},
        'smin': {'default': None, 'type': (float, int), 'alias': 's0'},
        'permax': {'default': None, 'type': (float, int), 'alias': 'pmax'},
        'fmax': {'default': None, 'type': (float, int)},
        'smax': {'default': None, 'type': (float, int)},
        'pcut': {'default': None, 'type': (float, int)},
        'fcut': {'default': None, 'type': (float, int)},
        'scut': {'default': None, 'type': (float, int)},
        'dj_lin': {'default': None, 'type': (float, int)},
        
        # Scale array (new parameter)
        'scales': {'default': None, 'transform': lambda v: np.asarray(v),
            'validate': lambda v: np.all(v > 0) and np.all(np.diff(v) > 0),
            'error_msg': 'scales must contain positive and strictly increasing values'},
        
        # Parameter limits
        'percentout': {'default': 0.02, 'type': (float, int),
            'validate': lambda v: 0 < v < 1,
            'error_msg': 'Invalid percentout value "{value}". Please specify a value in range (0,1)'},
        
        # Enum-like parameters
        'wname': {
            'default': 'mor',
            'transform': lambda v: str(v).lower(), 
            'validate': lambda v: v[:3] in VALID_OPTIONS['wname']},
        
        'scaletype': {
            'default': 'log', 
            'transform': lambda v: _parse_scaletype(v),
            'validate': lambda v: v in VALID_OPTIONS['scaletype']},
        
        'norm': {
            'default': None, 
            'transform': lambda v: _parse_norm(v),
            'validate': lambda v: v in VALID_OPTIONS['norm']},
        
        'pad': {
            'default': 'ref', 
            'transform': lambda v: str(v).lower()[:3],
            'validate': lambda v: v in VALID_OPTIONS['pad']},
        
        'padmode': {
            'default': 'b', 
            'transform': lambda v: str(v).lower()[0],
            'validate': lambda v: v in VALID_OPTIONS['padmode']},
        
        'coimethod': {
            'default': 'ana', 
            'transform': lambda v: str(v).lower()[:3],
            'validate': lambda v: v in VALID_OPTIONS['coimethod']},
        
        'coimetric': {
            'default': 'int', 
            'transform': lambda v: str(v).lower()[:3],
            'validate': lambda v: v in VALID_OPTIONS['coimetric']},
        
        'plotunits': {
            'default': None, 
            'transform': lambda v: _parse_plotunits(v),
            'validate': lambda v: v is None or v in VALID_OPTIONS['plotunits']},
        
        # Wavelet parameters
        's2': {'default': 1, 'type': (float, int),
            'validate': lambda v: v > 0,
            'error_msg': 'Invalid s2 value "{value}". Please specify a positive value'},
        'w0': {'default': 6, 'type': (float, int),
            'validate': lambda v: v > 0,
            'error_msg': 'Invalid w0 value "{value}". Please specify a positive value'},
        
        # Boolean parameters
        'figure': {'default': False, 'type': bool},
        'signif': {'default': False, 'type': bool},
        'coi': {'default': True, 'type': bool},
        'use_pyfftw': {'default': False, 'type': bool},
        'parallel_proc': {'default': False, 'type': bool},

        # Parallelization
        'workers': {'default': None, 'type': int},
        'chunk_size': {'default': None, 'type': int,
                    'validate': lambda v: v is None or v > 0,
                    'error_msg': 'Invalid chunk_size value "{value}". Please specify a positive integer'},
        
        # Data
        'xdata': {'default': None}
    }

    # Helper functions for parameter transformation
    def _parse_scaletype(v):
        """Parse scaletype from various input formats"""
        if isinstance(v, (list, tuple)):
            scale_values = [str(val).lower()[:3] for val in v]
            return 'bot' if {'log', 'lin'}.issubset(scale_values) else None
        return str(v).lower()[:3]
    
    def _parse_norm(v):
        """Parse norm to standard format"""
        v = str(v).upper()
        return f"L{v}" if len(v) == 1 else v
    
    def _parse_plotunits(v):
        """Parse plot units to standard format"""
        v = str(v).lower()
        return v[:-1] if v.endswith('s') else v

    # Initialize parameters with default values
    args = {name: spec['default'] for name, spec in param_specs.items()}
    
    # Handle aliases and alternative parameter names
    aliases = {}
    for name, spec in param_specs.items():
        if 'alias' in spec:
            aliases[spec['alias']] = name
    
    # Process all provided parameters
    processed_params = set()
    conflicts = []
    
    # First pass: process all provided parameters
    for key, value in kwargs.items():
        # Skip None values
        if value is None:
            continue

        # Handle aliases
        param_name = aliases.get(key, key)
        
        # Handle unknown parameters
        if param_name not in param_specs:
            raise ValueError(f'Input Argument "{key}" not recognized. Please refer to the documentation')
        
        processed_params.add(param_name)
        spec = param_specs[param_name]
        
        # Ensure parameter is not calculated internally
        if 'calculated' in spec:
            raise ValueError(spec.get('error_msg'))
    
        # Transform the value if a transformation function is provided
        if 'transform' in spec:
            try:
                value = spec['transform'](value)
            except Exception as e:
                raise ValueError(f"Failed to transform {param_name}: {e}")
        
        # Validate type if specified
        if 'type' in spec and not isinstance(value, spec['type']):
            type_names = [t.__name__ for t in spec['type']] if isinstance(spec['type'], tuple) else [spec['type'].__name__]
            raise TypeError(f"{param_name} must be {' or '.join(type_names)}, not {type(value).__name__}")
        
        # Validate value if a validation function is provided
        if 'validate' in spec and not spec['validate'](value):
            error_msg = spec.get('error_msg', f"Invalid value for {param_name}: {value}")
            if '{value}' in error_msg:
                error_msg = error_msg.format(value=value)
            raise ValueError(error_msg)
        
        # Store the validated value
        args[param_name] = value
    
    # Set norm based on wavelet type if not provided
    if args['norm'] is None:
        args['norm'] = 'L2' if args['wname'][:3] == 'gau' else 'L1'
    
    # Handle ts/fs exclusivity
    if args['ts'] is not None and args['fs'] is not None:
        conflicts.append("Sampling period (ts) and frequency (fs)")
    
    # If neither ts nor fs provided, use default
    if args['ts'] is None and args['fs'] is None:
        args['ts'] = 1
        args['dt'] = 1
        warnings.warn('Temporal resolution fixed to 1 (adimensionnal)', UserWarning, stacklevel=2)
    # Calculate dt from ts or fs
    elif args['ts'] is not None:
        args['dt'] = args['ts']
        if isinstance(args['ts'], timedelta):
            args['dt'], args['dt_fct'], _ = extract_duration_info(args['ts'])
    else:
        args['dt'] = 1 / args['fs']
    
    # Handle 'xdata' parameter validation relative to ts
    if args['xdata'] is not None:
        # If ts is a timedelta, xdata should be datetime
        if isinstance(args['ts'], timedelta) and not isinstance(args['xdata'][0], (datetime, np.datetime64, pd.Timestamp)):
            raise TypeError("When ts is a timedelta, 'xdata' must be a datetime array")
    
    # Check exclusivity groups
    exclusivity_groups = [
        ('permin', 'smin', 'fmax'),
        ('permax', 'smax', 'fmin'),
        ('pcut', 'scut', 'fcut')
    ]
    
    for group in exclusivity_groups:
        specified = [g for g in group if args[g] is not None]
        if len(specified) > 1:
            conflicts.append(', '.join(specified))
    
    # Calculate derived parameters
    if args['fmax'] is not None:
        args['permin'] = 1 / args['fmax']
    
    if args['fmin'] is not None:
        args['permax'] = 1 / args['fmin']
    
    if args['fcut'] is not None:
        args['pcut'] = 1 / args['fcut']
    
    # Handle dj_lin default
    if args['dj_lin'] is None and args['scales'] is None:
        args['dj_lin'] = args['dj']
        if args['scaletype'] == 'bot':
            warnings.warn('Linear spacing between scales set to dj. \nTo change this value, specify dj_lin', UserWarning, stacklevel=2)
    
    # Warning for L1 norm with Gaussian wavelets
    if args['norm'] == 'L1' and args['wname'][:3] == 'gau':
        warnings.warn("Fourier Factor (period to scale conversion) undefined for norm 'L1' using the Gaussian function. "
                      "Consider setting the norm to 'L2' for correct period conversions.", UserWarning, stacklevel=2)
    
    # Warning for w0 < 6
    if args['w0'] < 6:
        warnings.warn('Warning: The Morlet wavelet is admissible for w0>6. For lower values, additional terms cannot be neglected', UserWarning, stacklevel=2)
    
    # Report conflicts
    if conflicts:
        warning_msg = "Multiple values provided for exclusive parameters:\n" + \
                     "\n".join(f"- {c}" for c in conflicts) + \
                     "\nThe code will continue, but results may be unexpected."
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
    
    # Wavelet parameters
    wav_params = {
        'wname': args['wname'],
        'norm': args['norm'],
        's2': args['s2'],
        'w0': args['w0']
    }
    
    # Scale parameters
    if args['scales'] is not None:
        scale_param = args['scales']
    else:
        scale_param = {
            'permin': args['permin'],
            'smin': args['smin'],
            'permax': args['permax'],
            'smax': args['smax'],
            'pcut': args['pcut'],
            'scut': args['scut']
        }
    
    # Flags
    flags = {
        'coi': args['coi'],
        'signif': args['signif'],
        'figure': args['figure']
    }

    # WT computation
    processing = {
        'use_pyfftw': args['use_pyfftw'],
        'parallel_proc': args['parallel_proc'],
        'workers': args['workers'],
        'chunk_size': args['chunk_size']
    }

    # xdata
    xdata = args['xdata']
    
    # All other parameters
    excluded_keys = set(list(wav_params.keys()) + list(flags.keys()) + list(processing.keys()) + ['xdata'] + 
                        (['scales'] if args['scales'] is not None else list(scale_param.keys())) + 
                        ['fmin', 'fmax', 'fcut'])
    params = {k: v for k, v in args.items() if k not in excluded_keys}
    
    return params, wav_params, scale_param, flags, processing, xdata

def extract_duration_info(ts: timedelta) -> Tuple[float, Callable[[Union[int, float]], timedelta], str]:
    """
    Extract time unit information from a timedelta object.
    
    Parameters
    ----------
    ts : timedelta
        A time duration object.
        
    Returns
    -------
    dt : float
        The numerical value of the duration in the most appropriate time unit.
    fct : callable
        A function that converts a numerical value back to a timedelta object.
    tsformat : str
        A string identifier for the time unit used ('microsec', 'millisec', 'sec', 
        'min', 'hour', or 'day').
        
    Raises
    ------
    ValueError
        If ts is not an instance of timedelta.
        
    Notes
    -----
    This function automatically selects the most appropriate time unit based
    on the magnitude of the duration. It then provides conversion functions
    to work with this time unit.  
    """    
    # Get total duration in seconds 
    try:
        total_seconds = ts.total_seconds()
    except AttributeError:
        raise ValueError('Input must be an instance of timedelta.')
    
    # Define time unit conversion constants
    MICROSECS_PER_SEC = 1_000_000
    MILLISECS_PER_SEC = 1_000
    SECS_PER_MIN = 60
    SECS_PER_HOUR = 3600
    SECS_PER_DAY = 86400
    
    # Determine the appropriate time unit based on magnitude
    if total_seconds < 0.001:
        tsformat = 'microsec'
        dt = total_seconds * MICROSECS_PER_SEC
        fct = lambda x: timedelta(microseconds=x)
    elif total_seconds < 1:
        tsformat = 'millisec'
        dt = total_seconds * MILLISECS_PER_SEC
        fct = lambda x: timedelta(milliseconds=x)
    elif total_seconds < SECS_PER_MIN:
        tsformat = 'sec'
        dt = total_seconds
        fct = lambda x: timedelta(seconds=x)
    elif total_seconds < SECS_PER_HOUR:
        tsformat = 'min'
        dt = total_seconds / SECS_PER_MIN
        fct = lambda x: timedelta(minutes=x)
    elif total_seconds < SECS_PER_DAY:
        tsformat = 'hour'
        dt = total_seconds / SECS_PER_HOUR
        fct = lambda x: timedelta(hours=x)
    else:
        tsformat = 'day'
        dt = total_seconds / SECS_PER_DAY
        fct = lambda x: timedelta(days=x)
    
    return dt, fct, tsformat

def extract_value_duration(units: str) -> Tuple[Callable[[timedelta], Union[int, float]], str]:
    """
    Create conversion functions based on specified time units.
    
    Parameters
    ----------
    units : str
        The time unit to use for conversion (e.g., 'microsec', 'millisec', 'sec', 'min', 
        'hour', 'day').
        
    Returns
    -------
    fct : callable
        A function that converts a timedelta object to a numerical value in the specified unit.
    tsformat : str
        A standardized string identifier for the time unit.
        
    Raises
    ------
    ValueError
        If the specified unit is not recognized.
        
    Notes
    -----
    This function is the inverse operation of extract_duration_info. It creates
    functions to convert timedelta objects to numerical values in a specified time unit.
    
    Only the first three characters of the input units string are considered for matching,
    allowing for flexibility in how units are specified (e.g., 'sec', 'seconds', 'second').
    """
    # Define time unit conversion constants
    MICROSECS_PER_SEC = 1_000_000
    MILLISECS_PER_SEC = 1_000
    SECS_PER_MIN = 60
    SECS_PER_HOUR = 3600
    SECS_PER_DAY = 86400
    
    # Use only first three characters for matching
    units_key = units[:3].lower()
    
    # Conversion dictionary mapping unit prefixes to (conversion function, unit format string)
    unit_conversions = {
        'mic': ('microsec', lambda x: x.total_seconds() * MICROSECS_PER_SEC),
        'mil': ('millisec', lambda x: x.total_seconds() * MILLISECS_PER_SEC),
        'sec': ('sec', lambda x: x.total_seconds()),
        'min': ('min', lambda x: x.total_seconds() / SECS_PER_MIN),
        'hou': ('hour', lambda x: x.total_seconds() / SECS_PER_HOUR),
        'day': ('day', lambda x: x.total_seconds() / SECS_PER_DAY)
    }
    
    # Get the appropriate conversion
    if units_key in unit_conversions:
        tsformat, fct = unit_conversions[units_key]
    else:
        valid_units = ', '.join(['microsec', 'millisec', 'sec', 'min', 'hour', 'day'])
        raise ValueError(f'Unrecognized time unit "{units}". \nPlease use one of: {valid_units}')
    
    return fct, tsformat

def convert_timdelta_array(units: str, *args) -> Tuple[np.ndarray, ...]:
    """
    Convert multiple arrays of timedelta objects to numerical values.
    
    Parameters
    ----------
    units : str
        The time unit to use for conversion (e.g., 'microsec', 'millisec', 'sec', 'min', 
        'hour', 'day').
    *args : array_like
        One or more arrays containing timedelta objects to be converted.
        
    Returns
    -------
    tuple
        A tuple of numpy arrays containing the converted numerical values in the specified unit.
        The length of the tuple matches the number of input arrays.
        
    Notes
    -----
    This function acts as a batch processor for timedelta arrays, converting all elements
    in each array to numerical values using the specified time unit. 
    """
    # Get the conversion function for the specified unit
    fct_inv, _ = extract_value_duration(units)
    
    # Vectorize the conversion function for array processing
    fct_vectorized = np.vectorize(fct_inv)
    
    # Apply the vectorized function to each input array
    transformed_inputs = [fct_vectorized(array) for array in args]
    
    # Return results as a tuple (matches the number of input arrays)
    return tuple(transformed_inputs)

def get_as_timedelta(units: str, value: Union[int, float]) -> timedelta:
    """
    Convert a numerical value to a timedelta object based on the specified unit.
    
    Parameters
    ----------
    units : str
        The time unit to use for conversion. 
        Must be one of: 'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days'.
    value : int, float
        The numerical value to convert.
        
    Returns
    -------
    timedelta
        A timedelta object representing the converted value.
        
    Raises
    ------
    ValueError
        If the specified unit is not recognized.
        
    Notes
    -----
    This function is the inverse of extract_value_duration. It creates a timedelta object
    based on a numerical value and a specified time unit.
    
    Only the first three characters of the input units string are considered for matching,
    allowing for flexibility in how units are specified (e.g., 'sec', 'seconds', 'second').
    """
    # Get units in correct format
    # Use only first three characters for matching
    units_key = units[:3].lower()
    
    # Conversion dictionary mapping unit prefixes to (conversion function, unit format string)
    unit_conversions = {
        'mic': lambda x: timedelta(microseconds=x),
        'mil': lambda x: timedelta(milliseconds=x),
        'sec': lambda x: timedelta(seconds=x),
        'min': lambda x: timedelta(minutes=x),
        'hou': lambda x: timedelta(hours=x),
        'day': lambda x: timedelta(days=x)
    }
    
    # Convert the value to timedelta
    if units_key in unit_conversions:
        return unit_conversions[units_key](value)
    else:
        valid_units = ', '.join(['microsec', 'millisec', 'sec', 'min', 'hour', 'day'])
        raise ValueError(f'Unrecognized time unit "{units}". \nPlease use one of: {valid_units}')

def extract_signal_and_time(y: Union[ArrayLike, "pd.DataFrame", "pd.Series"], 
                            figure: bool = False, 
                            xdata: Optional[Union[ArrayLike, "pd.Series", "pd.DatetimeIndex", "DatetimeArray"]] = None
                            ) -> Tuple[np.ndarray, bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract signal data and time information from various input types.
    
    Parameters
    ----------
    y : array_like, pandas.Series or pandas.DataFrame
        The input signal data. Can be:
        - A numpy array (1D for single signal, 2D for multiple signals)
        - A pandas Series (containing a single signal)
        - A pandas DataFrame (columns contain signals, may include xdata)
        - A list/tuple of signals or array-like objects

    figure : bool, optional
        Whether to prepare data for plotting figures. If True, will attempt to
        extract x-axis information. Default is False.
    xdata : array_like, pandas.Series, pandas.DatetimeIndex, pandas.arrays.DatetimeArray, optional
        Custom time or x-axis data for the signal. Default is None.
        
    Returns
    -------
    y : numpy.ndarray
        The primary signal as a numpy array.
    xwt : bool
        Flag indicating whether a cross wavelet transform is possible (True if
        a second signal was found).
    y2 : numpy.ndarray or None
        The secondary signal as a numpy array, or None if no second signal exists.
    xdata : numpy.ndarray or None
        Time or x-axis data for plotting, or None if not available/requested.
        
    Raises
    ------
    TypeError
        If the input data type is not supported or if xdata is not a valid type.
        
    Notes
    -----
    This function handles multiple input formats and automatically extracts signals
    for wavelet analysis. It also identifies xdata information when available in
    pandas DataFrames (from columns named 'xdata' or 'time' or 'datetime', or from the index, if datetime).
    
    When 2D arrays are provided, the function determines whether signals are stored
    as rows or columns based on which dimension is smaller, assuming the smaller
    dimension represents different signals.
    """
    import pandas as pd
    from pandas.arrays import DatetimeArray
    from datetime import datetime
    
    # Initialize variables
    y2 = None  # Second signal
    xwt = False  # Flag for cross wavelet transform
    msg = 'arrays'  # Message format identifier
    msg2 = ''  # Additional message
    nel = 1  # Number of elements/signals
    
    # Process different input types           
    if isinstance(y, pd.DataFrame):
        # Handle pandas DataFrame
        if y.empty:
            raise ValueError('y must contain at least one signal (DataFrame is empty)')
        cols = y.columns
        
        # Try to extract time information if plotting is requested
        if figure and (xdata is None):
            columns_lower = [col.lower() for col in cols]

            # Look for xdata column
            if 'xdata' in columns_lower:
                xdata = y[cols[columns_lower.index('xdata')]]
                msg2 = '\nUsing xdata column found in DataFrame as xdata'
            # Look for time columns with standard names
            elif 'time' in columns_lower:
                xdata = y[cols[columns_lower.index('time')]]
                msg2 = '\nUsing time column found in DataFrame as x-axis for the scalogram'
            elif 'datetime' in columns_lower:
                xdata = y[cols[columns_lower.index('datetime')]]
                msg2 = '\nUsing datetime column found in DataFrame as x-axis for the scalogram'
            # If no time column, check if index is datetime
            elif pd.api.types.is_datetime64_any_dtype(y.index):
                xdata = y.index
                msg2 = '\nUsing DataFrame index as x-axis for the scalogram'
        
        # Get signal data from DataFrame
        nel = len(cols)
        if nel > 1:
            xwt = True
            y2 = y.iloc[:, 1].to_numpy()
        y = y.iloc[:, 0].to_numpy()

    elif isinstance(y, pd.Series):
        # Handle pandas Series
        if y.empty:
            raise ValueError('y must contain at least one signal (Series is empty)')
        
        # Check for datetime index if figure=True
        if figure and xdata is None:
            if pd.api.types.is_datetime64_any_dtype(y.index):
                xdata = y.index  # Use the Series index if it's datetime
                msg2 = '\nUsing Series index as x-axis for the scalogram'
        # Convert to numpy array
        y = y.to_numpy()
    
    elif isinstance(y, np.ndarray):
        # Handle numpy arrays
        if y.ndim > 1:  # Multi-dimensional array
            # Determine orientation (rows vs columns) by finding smallest dimension
            nmin = np.argmin(y.shape)
            nel = y.shape[nmin]
            
            if nmin == 0:  # Signals stored as rows
                msg = 'lines'
                y_primary = y[0, :]
                
                # Extract second signal if available
                if nel > 1:
                    y2 = y[1, :]
                    xwt = True
                    
            else:  # Signals stored as columns
                msg = 'columns'
                y_primary = y[:, 0]
                
                # Extract second signal if available
                if nel > 1:
                    y2 = y[:, 1]
                    xwt = True
                    
            y = y_primary
        else:
            nel = 1
    
    elif isinstance(y, (tuple, list)) or hasattr(y, "__array__"):
        # Handle lists, tuples, or objects with __array__
        if len(y) == 0:
            raise ValueError('y must contain at least one signal')
        '''
        # Check for array-like objects in the list
        def count_arrays(y):
            return sum(1 for elem in y if isinstance(elem, np.ndarray) or hasattr(elem, "__array__"))
        # Count the number of array-like objects
        nel = count_arrays(y)
        # Convert to NumPy array while keeping object dtype
        y = np.asarray(y, dtype=object)
        if nel > 1:
            xwt = True
            y2 = np.asarray(y[1])  # Second signal as NumPy array

        y = np.asarray(y[0])  # First signal as NumPy array
        '''
        # Convert to NumPy array while keeping object dtype
        y = np.asarray(y, dtype=object)
        # Convert each element in the list/tuple to a NumPy array (if it's array-like)
        array_like = [np.asarray(elem) for elem in y if isinstance(elem, (np.ndarray, list, tuple)) or hasattr(elem, "__array__")]
        # Count the number of array-like objects
        nel = len(array_like)
        # Extract the second signal if provided
        if nel > 1:
            xwt = True
            y2 = array_like[1]
        # Extract the first signal
        y = array_like[0]

    else:
        # Unsupported input type
        raise TypeError(f'y must be a numeric array, not a {type(y).__name__}')
    
    # Inform user if more than 2 signals were provided but only 2 are used
    if nel > 2:
        import warnings
        warnings.warn(f'Only first two {msg} of y considered as signals. Other entries are ignored.', UserWarning, stacklevel=2)
    # Inform user that x-axis data was found
    if msg2:
        import warnings
        warnings.warn(msg2, UserWarning, stacklevel=2)
    
    # Process x-axis data for plotting
    if figure and (xdata is not None):
        # Convert numeric input types to numpy array
        if not isinstance(xdata, (np.ndarray, pd.Series, pd.DatetimeIndex, DatetimeArray)):
            try:
                xdata = np.asarray(xdata)
            except ValueError:
                raise ValueError(f'xdata must be convertible to a numpy array, but found {type(xdata).__name__}')
        
        # Validate xdata length and data type
        if len(xdata) == 0:
            raise ValueError('xdata cannot be empty')
        
        # Check if elements are datetime or numeric
        if isinstance(xdata, (np.ndarray, pd.Series)) and not isinstance(xdata[0], (datetime, np.datetime64, pd.Timestamp, int, float)):
            raise TypeError("xdata must contain datetime or numeric values")

    return y, xwt, y2, xdata

def sig_length_and_padding(n: Union[int, float], 
                           pad: str = 'none', 
                           padmode: str = 'b'
                           ) -> Tuple[Optional[Callable[[np.ndarray], np.ndarray]], Optional[Callable[[np.ndarray], np.ndarray]], int]:
    """
    Provide functions to apply and remove padding.
    
    Parameters
    ----------
    n : int or float
        Length of the input signal to be padded.
    pad : str, optional (default='none')
        Padding method to use. Options are:
        - 'none': No padding
        - 'zpd': Zero padding
        - 'sym': Symmetric padding (mirror reflection without repeating edge values)
        - 'ref': Reflection padding (mirror reflection with repeating edge values)
        - 'per': Periodic padding (repeating the signal)

    padmode : str, optional (default='b')
        Where to apply padding. Options are:
        - 'r': Right padding only
        - 'l': Left padding only
        - 'b': Both left and right padding

    Returns
    -------
    apply_padding : callable or None
        A function that applies the padding to a signal.
        Set to None if no padding is desired.
    remove_padding : callable or None
        A function that removes the padding from wavelet coefficients.
        Set to None if no padding was applied.
    n_ext : int
        Padded signal length.
        
    Notes
    -----
    This function pads the signal to a length that is a power of 2, which
    improves the efficiency of FFT-based wavelet transforms. The padding
    approach affects edge behavior in the wavelet transform.
    
    For FFT efficiency, the function finds the next power of 2 that is
    greater than or equal to the signal length, then define padding functions accordingly.

    The returned remove_padding function is designed to be applied to
    wavelet coefficient arrays, targeting the position/time dimension.
    """
    # Original signal length
    n_ext = n
    # Initialize padding functions
    apply_padding = None
    remove_padding = None

    # Apply padding if requested
    if pad.lower() != 'none':
        # Find the base-2 logarithm of n and round up to next power of 2
        base = int(np.log2(n) + 0.4999)
        
        # Apply padding based on the selected mode
        if padmode == 'r':  # Right padding
            # Calculate extension length
            ext = 2**(base + 1) - n
            # Length after padding
            n_ext += ext
            # Function to remove padding later
            remove_padding = lambda x: x[:, :n]
            
            # Apply the specified padding method
            if pad == 'zpd':      # Zero padding
                apply_padding = lambda x: np.concatenate((x, np.zeros(ext)))
            elif pad == 'sym':    # Symmetric padding
                apply_padding = lambda x: np.concatenate((x, x[::-1][:ext]))
            elif pad == 'ref':    # Reflection padding
                apply_padding = lambda x: np.concatenate((x, x[::-1][1:ext + 1]))
            elif pad == 'per':    # Periodic padding
                apply_padding = lambda x: np.concatenate((x, x[:ext]))
                
        elif padmode == 'l':  # Left padding
            # Calculate extension length
            ext = 2**(base + 1) - n
            # Length after padding
            n_ext += ext
            # Function to remove padding later
            remove_padding = lambda x: x[:, -n:]
            
            # Apply the specified padding method
            if pad == 'zpd':      # Zero padding
                apply_padding = lambda x: np.concatenate((np.zeros(ext), x))
            elif pad == 'sym':    # Symmetric padding
                apply_padding = lambda x: np.concatenate((x[:ext][::-1], x))
            elif pad == 'ref':    # Reflection padding
                apply_padding = lambda x: np.concatenate((x[1:ext + 1][::-1], x))
            elif pad == 'per':    # Periodic padding
                apply_padding = lambda x: np.concatenate((x[-ext:], x))
                
        elif padmode == 'b':  # Both sides padding
            # Calculate extension length (half on each side)
            ext = 2**base - n // 2
            # Length after padding
            n_ext += ext * 2
            # Left extension length (ceiling) and right extension length (floor)
            left_ext = int(np.ceil(ext))
            right_ext = int(np.floor(ext))
            # Function to remove padding later
            remove_padding = lambda x: x[:, left_ext:left_ext + n]
            
            # Apply the specified padding method
            if pad == 'zpd':      # Zero padding
                apply_padding = lambda x: np.concatenate((np.zeros(left_ext), x, np.zeros(right_ext)))
            elif pad == 'sym':    # Symmetric padding
                apply_padding = lambda x: np.concatenate((x[:left_ext][::-1], x, x[::-1][:right_ext]))
            elif pad == 'ref':    # Reflection padding
                apply_padding = lambda x: np.concatenate((x[1:left_ext + 1][::-1], x, x[::-1][1:right_ext + 1]))
            elif pad == 'per':    # Periodic padding
                apply_padding = lambda x: np.concatenate((x[-left_ext:], x, x[:right_ext]))

    return apply_padding, remove_padding, n_ext