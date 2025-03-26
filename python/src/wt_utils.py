"""
Created on Fri Feb 14 2025

@author: Jonathan Bitton
"""
# Standard library imports
from typing import Callable, Optional, Tuple, Union, Dict, Any, TYPE_CHECKING

# Third-party imports
import numpy as np
from numpy.typing import ArrayLike

# Variable type hints
if TYPE_CHECKING:
    import pandas as pd
    from pandas.arrays import DatetimeArray
    from datetime import timedelta
    import matplotlib

def define_wavelet(wname: str, 
                   norm: Optional[str] = None, 
                   s2: float = 1, 
                   w0: float = 6) -> Dict[str, Any]:
    """
    Define a wavelet dictionary with properties for wavelet transform calculations.
    
    Parameters
    ----------
    wname : str (default='mor')
        Wavelet name. Supported types include (for additional formats, see notes section):
        - 'mor' or 'morlet': Morlet wavelet (complex)
        - 'mex' or 'mexhat': Mexican hat wavelet (equivalent to dog2) 
        - 'dog{m}': Derivative of Gaussian of order m
        - 'gau' or 'gauss': Gaussian function (not a true wavelet)
        - 'haa' or 'haar': Haar wavelet
    norm : str, optional (default='L2' for Gaussian, 'L1' for others)
        Normalization type: 'L1' or 'L2'.
    s2 : int or float, optional (default=1)
        Variance parameter for Gaussian-based wavelets.
    w0 : int or float, optional (default=6)
        Central angular frequency parameter for Morlet wavelet.
        
    Returns
    -------
    dict
        Dictionary containing wavelet properties:
        - 'name': Normalized wavelet name
        - 'fourier_fac': Fourier factor (ratio of period to scale)
        - 'norm': Normalization type used
        - 'norm_fac': Normalization factor
        - 'renorm_fac': Re-normalization factor (L1 to L2)
        - 'params': Dictionary of wavelet parameters
        - 'is_complex': Boolean indicating if wavelet is complex-valued
        - 'psi': Function that computes the wavelet in time domain
        - 'psi_fft': Function that computes the wavelet in frequency domain
        - 'support': Function that computes the wavelet support
        
    Notes
    -----
    Special wavelet specifications:
    - Morlet: 'mor{w0}-{s2}' sets central frequency and variance of Gaussian enveloppe
    - Gaussian: 'gau{s2}' sets variance
    - DOG: 'dog{m}-{s2}' sets derivative order and variance of Gaussian
    - Mexican hat: 'mex{s2}' sets variance of Gaussian (equivalent to 'dog2-{s2}')
    
    The Morlet wavelet is only admissible for w0 > 6. The function will 
    issue a warning if w0 < 6 is specified.
    
    Examples
    --------
    >>> wavelet = define_wavelet('mor8-1.5')  # Morlet with w0=8, s2=1.5
    >>> wavelet = define_wavelet('dog2')      # Mexican hat (2nd derivative of Gaussian)
    >>> wavelet = define_wavelet('haar', norm='L2')  # Haar wavelet with L2 normalization
    """
    import scipy as sp
    import re

    # Format wavelet name
    wname = str(wname).lower()
    wavname = wname[:3]   

    # Process different wavelet types and extract parameters
    if wavname == 'dog':  # Derivative of Gaussian
        # Parse format: dog{m}-{s2} or dog{m}
        matchd = re.search(r'^dog([\d.]+)-([\d.]+)$', wname)
        if matchd:
            m = float(matchd.group(1))
            s2 = float(matchd.group(2))
        elif re.match(r'^dog[\d.]+$', wname):
            m = float(wname[3:])
        else: 
            raise ValueError(f'Invalid DOG wavelet name: "{wname}". '
                             'Expected format: "dogM" or "dogM-S2" where M is derivation order and S2 is variance')
        
        # Validate derivation order
        if not m.is_integer():
            raise ValueError(f'Derivation order of DOG wavelet ({m}) must be an integer')
        
        m = int(m)
        if m == 0:
            wavname = 'gau'  # 0th derivative = Gaussian
        else:
            wavname = f'dog{m}'
    
    elif wavname == 'mex':  # Mexican hat (2nd derivative of Gaussian)
        wavname = 'dog'
        m = 2
        # Parse format: mex{s2} or mex
        matchd = re.search(r'[\d.]+$', wname)
        if matchd:
            s2 = float(matchd.group())
            
    elif wavname == 'gau':  # Gaussian
        # Parse format: gau{s2} or gau
        matchd = re.search(r'[\d.]+$', wname)
        if matchd:
            s2 = float(matchd.group())
    
    elif wavname == 'mor':  # Morlet
        # Parse format: mor{w0}-{s2} or mor
        matchd = re.search(r'([\d.]+)-([\d.]+)$', wname)
        if matchd:
            w0 = float(matchd.group(1))
            s2 = float(matchd.group(2))
        else:
            matchd = re.search(r'[\d.]+$', wname)
            if matchd:
                w0 = float(matchd.group())
    
    # Set complex flag and default normalization
    is_complex = wavname == 'mor'  # Only Morlet is complex
    NormL1 = 2  # Default L1 normalization factor
    # Set or validate norm
    if norm is None:
        norm = 'L2' if wavname == 'gau' else 'L1'  # Default normalization
    else:
        norm = str(norm).upper()
        if len(norm) == 1:
            norm_i = norm
            norm = f'L{norm}'
        if norm not in ['L1', 'L2']:
            if norm_i in locals():
                norm = norm_i
            raise ValueError(f'Normalization method "{norm}" invalid. \nPlease select an option among L1 or L2')

    
    # Configure wavelet parameters based on type
    if wavname == 'mor': # Morlet wavelet
        # Check admissibility condition
        if w0 < 6:
            import warnings
            warnings.warn('Warning: The Morlet wavelet is admissible for w0 > 6. '
                        'For lower values, additional terms cannot be neglected', UserWarning, stacklevel=2)
        
        # Store parameters
        params = {'s2': s2, 'w0': w0}
        
        # Set normalization factors based on norm type
        if norm == 'L1':
            # Fourier factor (ratio period/scale)
            fourier_fac = 2 * np.pi / w0
            # Normalization constant
            cst = NormL1
        else:  # L2 norm
            # Fourier factor (ratio period/scale)
            fourier_fac = 4 * np.pi * np.sqrt(s2) / (np.sqrt(s2) * w0 + np.sqrt(2 + s2 * w0**2))
            # Normalization constant
            cst = (4 * np.pi * s2)**(1/4)
            
        # Re-normalization constant (L1 to L2)
        cst2 = (4 * np.pi * s2)**(1/4) / cst
        
        # Define wavelet function in time domain
        def psi(x):
            norm_factor = cst/np.sqrt(2*np.pi*s2)
            complex_morl = norm_factor * np.exp(1j * w0 * x) * np.exp(-0.5 * x**2 / s2)
            return complex_morl
        
        # Define wavelet function in frequency domain
        def psi_fft(w):
            return cst * np.exp(- s2 * (w - w0)**2 / 2) * (w > 0)
        
        # Define function to compute wavelet support
        def support(scale=1, percentout=0.1, metric='int'):
            from scipy.special import erfinv

            # Validate percentout
            if not isinstance(percentout, float) or not (0 < percentout < 1):
                raise ValueError("Support computation: percentout must be a float in range (0, 1)")
            
            # Parse metric
            metric = str(metric)[:3].lower()

            # Compute support based on metric
            if metric == 'int':  # Absolute integral (L1 norm)
                # Analytical solution for 1-percentout area
                I = erfinv(2 * (0.5 - percentout / 2)) * np.sqrt(2 * s2)
            elif metric == 'ene':  # Energy (L2 norm)
                # Analytical solution for 1-percentout energy
                I = erfinv(2 * (0.5 - percentout / 2)) * np.sqrt(s2)
            else:
                raise ValueError(f"Support computation: Metric '{metric}' invalid. "
                                "Please use 'integral' or 'energy'")

            return 2 * I * scale  # Full support (both sides)   
        
    elif wavname == 'gau': # Gaussian (filtering, not a true wavelet)
        # Store parameters
        params = {'s2': s2}
        
        # Fourier factor (ratio period/scale)
        fourier_fac = 2 * np.pi * np.sqrt(s2 * 2)

        # Normalization constant
        if norm == 'L1':
            cst = NormL1
        else:
            cst = np.sqrt(2 * np.sqrt(np.pi * s2))

        # Re-normalization constant (L1 to L2)
        cst2 = np.sqrt(2 * np.sqrt(np.pi * s2)) / cst
        
        # Define Gaussian function in time domain
        def psi(x):
            norm_factor = cst / np.sqrt(2 * np.pi * s2)
            return norm_factor * np.exp(-0.5 * x**2 / s2)
        
        # Define Gaussian function in frequency domain
        def psi_fft(w):
            return cst * np.exp(- s2 * w**2 / 2)
        
        # Define function to compute Gaussian support
        def support(scale=1, percentout=0.1, metric='int'):
            from scipy.special import erfinv
            # Validate percentout
            if not isinstance(percentout, float) or not (0 < percentout < 1):
                raise ValueError("Support computation: percentout must be a float in range (0, 1)")
            
            # Parse metric
            metric = str(metric)[:3].lower()

            # Compute support based on metric
            if metric == 'int':  # Absolute integral (L1 norm)
                # Analytical solution for 1-percentout area
                I = erfinv(2 * (0.5 - percentout / 2)) * np.sqrt(2 * s2)
            elif metric == 'ene':  # Energy (L2 norm)
                # Analytical solution for 1-percentout energy
                I = erfinv(2 * (0.5 - percentout / 2)) * np.sqrt(s2)
            else:
                raise ValueError(f"Support computation: Metric '{metric}' invalid. "
                                "Please use 'integral' or 'energy'")

            return 2 * I * scale  # Full support (both sides)
        
    elif wavname[:3] == 'dog': # Derivatives Of Gaussian
        # Store parameters
        params = {'s2': s2, 'm': m}
        
        if norm == 'L1':
            # Fourier factor (ratio period/scale)
            fourier_fac = 2 * np.pi * np.sqrt(s2) / np.sqrt(m)
            # Normalization constant
            val = Retrieve_DOG_integral(m, s2)
            cst = -(1j)**m * NormL1 / val * np.sqrt(2 * np.pi * s2)
        else:
            # Fourier factor (ratio period/scale)
            fourier_fac = 2 * np.pi * np.sqrt(s2) / np.sqrt(m + 1/2)
            # Normalization constant
            cst = -(1j)**m * np.sqrt(s2**(m - 0.5) / sp.special.gamma(m + 0.5)) * np.sqrt(2 * np.pi * s2)
        
        # Re-normalization constant (L1 to L2)
        cst2 = np.sqrt(s2**(m - 0.5) / sp.special.gamma(m + 0.5)) * np.sqrt(2 * np.pi * s2) / abs(cst)

        # For even derivatives, normalization constant is real
        if m % 2 == 0:
            cst = cst.real    
        
        # Define DOG wavelet function in time domain
        def psi(x):
            norm_factor = abs(cst) * (-1) / np.sqrt(2 * np.pi * s2) #*(-1)**m * (-1)**(m+1) = *(-1)
            hermite_poly = sp.special.hermite(m)
            return  norm_factor * hermite_poly(x / np.sqrt(2 * s2)) / np.sqrt(s2 * 2)**(m) * np.exp(-0.5 * x**2 / s2)

        # Define DOG wavelet function in frequency domain
        def psi_fft(w):
            return cst * np.exp(- s2 * w**2 / 2) * w**m
        
        # Define function to compute DOG support
        def support(scale=1, percentout=0.1, metric='int'):
            # Use COI function to compute support
            coival = dog_coi_factor(m, s2, fourier_fac, metric, percentout / 2) # Exploiting COI function
            I = fourier_fac / coival  # Convert COI factor to support
            return 2 * I * scale  # Full support (both sides)

    elif wavname == 'haa': # Haar wavelet
        # Store parameters (none specific to Haar)
        params = {}

        if norm == 'L1':
            # Fourier factor (ratio period/scale)
            fourier_fac = 2 * np.pi / 4.6622447
            # Normalization constant
            cst = NormL1
        else:
            # Fourier factor (ratio period/scale)
            fourier_fac = 2 * np.pi / 5.5729963
            # Normalization constant
            cst = 1
        
        # Re-normalization constant (L1 to L2)
        cst2 = 1 / cst
        
        # Define Haar wavelet function in time domain
        def psi(x):
            psi = np.zeros_like(x, dtype=float)
            psi[(x >= -0.5) & (x < 0)] = cst
            psi[(x >= 0) & (x < 0.5)] = -cst
            return psi
        
        # Define Haar wavelet function in frequency domain
        def psi_fft(w):
            psi_values = np.zeros_like(w, dtype=np.complex128)
            mask = w != 0  # Avoid division by zero
            psi_values[mask] = -4 * cst * 1j * (np.sin(w[mask] / 4))**2 / w[mask]
            return psi_values
        
        # Define function to compute Haar support
        def support(scale=1, percentout=1, metric='int'):
            # Haar wavelet has compact support by definition
            return scale

    else:
        raise ValueError(f'Wavelet "{wname}" not supported. Available options: '
                        'morlet, mexhat, dogX (X=derivation order), gauss, haar')

    return {
        'name': wavname,
        'fourier_fac': fourier_fac,
        'norm': norm,
        'norm_fac': cst,
        'renorm_fac': cst2,
        'params': params,
        'is_complex': is_complex,
        'psi': psi,
        'psi_fft': psi_fft,
        'support': support
    }


def define_scales(fourier_fac: float, 
                  dt: Union[int, float], 
                  dj: Union[int, float], 
                  n: int, 
                  scaletype: str = 'log', 
                  dj_lin: Optional[Union[int, float]] = None, 
                  s0: Optional[Union[int, float]] = None, 
                  permin: Optional[Union[int, float]] = None, 
                  permax: Optional[Union[int, float]] = None, 
                  smax: Optional[Union[int, float]] = None, 
                  pcut: Optional[Union[int, float]] = None, 
                  scut: Optional[Union[int, float]] = None) -> Tuple[np.ndarray, Optional[float]]:
    """
    Define scales for wavelet transform.
    
    Parameters
    ----------
    fourier_fac : float
        Fourier factor (ratio of period to scale).
    dt : float, int
        Time step.
    dj : float, int
        Scale resolution for logarithmic scales.
    n : int
        Signal length.
    scaletype : str, optional (default='log')
        Scale spacing type: 'log', 'lin', or 'both' (mixed).
    dj_lin : float, int or None, optional (default=None)
        Scale resolution for linear scales when scaletype='bot'.
    s0 : float, int or None, optional (default=None)
        Smallest scale to analyze. If None, calculated from permin.
    permin : float, int or None, optional (default=None)
        Smallest period to analyze. If None, defaults to 2*dt.
    permax : float, int or None, optional (default=None)
        Largest period to analyze. If None, defaults to n*dt.
    smax : float, int or None, optional (default=None)
        Largest scale to analyze. If None, calculated from permax.
    pcut : float, int or None, optional (default=None)
        Period value where scaletype transitions from log to linear.
    scut : float, int or None, optional (default=None)
        Scale value where scaletype transitions from log to linear.
        
    Returns
    -------
    scales : numpy.ndarray
        Array of scales.
    fper : numpy.ndarray
        Periods or frequencies corresponding to scales.
    periods : numpy.ndarray
        Unitless periods.
    scut : float or None
        Transition scale (only relevant for 'both' scaletype).
        
    Notes
    -----
    Only one of s0/permin and smax/permax should be specified.
    For 'both' scaletype, one of pcut or scut must be specified.
    """
    import warnings

    # Lowest scale
    if s0 is None:
        if permin is None:
            permin = 2 * dt  # Default
        else:
            # fmax or permin specified
            if permin < 2 * dt:
                warnings.warn('permin(fmax) does not respect the Nyquist frequency. \n A value of 2*ts(fs/2) is recommended', UserWarning, stacklevel=2)
        s0 = permin / fourier_fac  # Converting periods to scales using the Fourier factor
    elif s0 * fourier_fac < 2 * dt:
        warnings.warn('smin does not respect the Nyquist frequency. \n A value of 2*ts(fs/2) for permin(fmax) is recommended', UserWarning, stacklevel=2)

    # Highest scale
    if smax is None:
        if permax is None:
            permax = n * dt  # Default
        else:
            if permax > n * dt:
                warnings.warn('permax(fmin) exceeds the signal length', UserWarning, stacklevel=2)
        smax = permax / fourier_fac  # Converting periods to scales using the Fourier factor
    elif smax * fourier_fac > n * dt:
        warnings.warn('smax exceeds the signal length', UserWarning, stacklevel=2)

    # Lowest scale analyzed must be inferior to the highest and positive
    if smax < s0:
        raise ValueError('permin>permax, smin>smax or fmax>fmin are inadmissible')
    if s0 < 0:
        raise ValueError(f'Minimum scale {s0} is negative. Only positive values are admitted')        
    
    if scaletype=='lin':
        # Scales for which wavelets are computed
        scales = np.arange(s0, smax, dj)
    else:
        # Maximal Index for scales (log computation)
        J = int(np.log2(smax / s0) / dj)
        # Scales for which wavelets are computed (log)
        scales = s0 * 2**(np.arange(0, J+1) * dj)
        # If transition
        if scaletype[:3] == 'bot':
            if scut is None:
                if pcut is None:
                    raise ValueError('at least one argument among scut, pcut or fcut must be set to define the scale, period or frequency for transition between log and linear scales')
                scut = pcut / fourier_fac 
            if scut < s0 or scut > smax:
                raise ValueError('Transition scale (or period/frequency) must be included in scale (or period/frequency) bounds')
            scut_idx = np.abs(scales - scut).argmin()
            scut = scales[scut_idx]
            # Create the first part: all values before and including scut
            scales_log = scales[:scut_idx + 1]
            
            # Create the second part: values starting from scut
            scales_lin = np.arange(scales_log[-1], smax, dj_lin)
            
            # Concatenate the two arrays
            scales = np.concatenate((scales_log, scales_lin[1:]))
    
    return scales, scut

def convert_scales(scales: np.ndarray, 
                   fourier_fac: float,
                   dt_fct: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert wavelet scales to periods or frequencies.
    
    Parameters
    ----------
    scales : numpy.ndarray
        Array of scales used in the wavelet transform.
    dt_fct : callable or None
        Function to apply for time unit conversion (for timedelta objects).
    fourier_fac : float
        Fourier factor for the specific wavelet, used to convert between
        scales and periods.
        
    Returns
    -------
    periods : numpy.ndarray
        Unitless periods.
    frequencies : numpy.ndarray
        Frequencies corresponding to the input scales.
    periods_with_units : numpy.ndarray
        Periods with units returned by dt_fct. If dt_fct is None, returns None.
    
    Notes
    -----
    The conversion follows these rules:
    1. Convert scales to periods by multiplying by the Fourier factor
    2. If dt_fct is provided (for time units), apply the function to periods
    3. Convert periods to frequencies
    """
    # Convert scales to periods using the Fourier factor
    periods = fourier_fac * scales  # periods analyzed (unitless)
    
    # Apply unit conversion if needed
    if dt_fct is not None:  # add units if 'ts' is a duration
        vectorized_fct = np.vectorize(dt_fct)
        periods_with_units = vectorized_fct(periods)
    else:
        periods_with_units = None

    # Convert periods to frequencies
    frequencies = 1.0 / periods
        
    return periods, frequencies, periods_with_units

def get_dj(scales: np.ndarray, base: Union[int, float] = 2) -> Tuple[Optional[float], Optional[str]]:
    """
    Detect scale type (linear or logarithmic) and calculate frequency resolution.
    
    Parameters
    ----------
    scales : numpy.ndarray
        Array of scales used in the wavelet transform.
    base : int or float
        Base of the logarithm for logarithmic scales (default is 2).
        
    Returns
    -------
    dj : float or None
        Frequency resolution parameter:
        - For linear scales: the constant difference between consecutive scales
        - For logarithmic scales: the logarithmic spacing parameter
        - None if the scale type cannot be determined

    scaletype : str or None
        Type of scale spacing:
        - 'lin' for linear spacing
        - 'log' for logarithmic spacing
        - None if the scale type cannot be determined
        
    Notes
    -----
    This function analyzes the provided scales to determine if they follow
    a linear or logarithmic spacing pattern. It uses second differences to
    detect constant increments (linear) or constant ratios (logarithmic).
    
    The tolerance for determining if scales are linear or logarithmic is 1e-12.
    
    For logarithmic scales, dj represents the parameter in the formula:
    s_j = s_0 * 2^(j*dj) where j is the scale index. If another base than 2 is used, 
    the value returned for dj is scaled by log2(base).
    """
    import warnings
    # Check if scales array has enough elements for analysis
    if len(scales) < 3:
        warnings.warn('Warning: Need at least 3 scale points to determine scale type', UserWarning)
        return None, None
        
    # Compute tolerance-based tests for linear and logarithmic spacing
    tolerance = 1e-12
    
    # Test for linear spacing (constant first differences)
    if np.all(np.abs(np.diff(np.diff(scales))) < tolerance):
        dj = scales[1] - scales[0]  # Step size
        scaletype = 'lin'
    # Test for logarithmic spacing (constant ratio between consecutive scales)
    elif np.all(np.abs(np.diff(np.diff(np.log2(scales)))) < tolerance):
        # Formula: s_j = s_0 * 2^(j*dj)
        dj = np.log2(scales[-1] / scales[0]) / (len(scales) - 1)
        scaletype = 'log'
        if base != 2:
            dj /= np.log2(base)
    else:
        warnings.warn('Scales are not linear or logarithmic. Unable to deduce dj', UserWarning)
        dj = None
        scaletype = None
        
    return dj, scaletype

def Retrieve_DOG_integral(m: int, s2: Union[int, float]) -> float:
    """
    Calculate the normalization integral for the Derivative of Gaussian (DOG) wavelet.
    
    Parameters
    ----------
    m : int
        Order of the DOG wavelet (number of derivatives).
    s2 : int or float
        Variance of the Gaussian function.
        
    Returns
    -------
    val : float
        Value of the normalization integral required for proper wavelet normalization.
        
    Notes
    -----
    This function calculates the normalization constant for the m-th order
    Derivative of Gaussian (DOG) wavelet, defined as:
    
    ∫|d^m/dx^m exp(-x^2/(2*s^2))| dx from -∞ to ∞
    
    For m < 6, pre-computed analytical expressions are used for efficiency.
    For m >= 6, numerical integration is performed using SymPy and SciPy's quad.
    
    The scale factor s2 represents the variance of the Gaussian function.
    """
    # For common orders (m < 6), use pre-computed analytical expressions for efficiency
    if m < 6:
        # Analytical expressions for common DOG wavelet orders
        # These have been mathematically derived to avoid numerical integration
        analytical_expressions = {
            1: 2,
            2: 4 / np.sqrt(np.exp(1)),
            3: 8 / np.sqrt(np.exp(3)) + 2,
            4: 4 * np.sqrt(6) / np.exp(1.5 + np.sqrt(1.5)) * (np.sqrt(3 + np.sqrt(6)) + np.sqrt(3 - np.sqrt(6)) * np.exp(np.sqrt(6))),
            5: 2 / np.exp(2.5 + np.sqrt(2.5)) * (16 + 8 * np.sqrt(10) + (8 * np.sqrt(10) - 16) * np.exp(np.sqrt(10))) + 6
        }
        
        # Apply scaling based on s2
        val = analytical_expressions[m] / s2**((m - 1) / 2)
    else:
        # For higher orders, use numerical integration
        import sympy as smp
        from scipy.integrate import quad, IntegrationWarning
        import warnings
        
        # Suppress integration warnings since we're handling infinite integrals
        warnings.filterwarnings('ignore', category=IntegrationWarning)
        
        # Symbolically define the function and its derivatives
        xvar = smp.symbols('xvar')
        expr = smp.diff(smp.exp(-xvar**2 / (2 * s2)), xvar, m)
        
        # Convert to a numerical function for integration
        func = smp.lambdify(xvar, smp.Abs(expr), 'numpy')
        
        # Perform numerical integration
        val, _ = quad(func, -np.inf, np.inf, epsrel=1e-15, epsabs=1e-15, limit=5000)
        
    return val

def compute_psi(n: int, 
                dt: Union[int, float], 
                wavelet: Dict[str, Any], 
                scales: np.ndarray) -> np.ndarray:
    """
    Compute the scaled wavelet in Fourier space for all scales.
    
    Parameters
    ----------
    n : int
        Length of the signal (after padding if applicable).
    dt : int or float
        X-axis step between consecutive samples.
    wavelet : dict
        Dictionary containing wavelet properties including:
        - 'psi_fft': Function to compute the wavelet in Fourier space
        - 'norm': Normalization method ('L1' or 'L2')
    scales : numpy.ndarray
        Array of scales at which to compute the wavelet transform.
        
    Returns
    -------
    psiscaled_ft : numpy.ndarray
        Scaled wavelets in Fourier space with shape (n_scales, signal_length).
        
    Notes
    -----
    This function computes the Fourier transform of the wavelet at each scale,
    which is later used for convolution via FFT multiplication.
    
    For 'L2' normalization, an additional scaling factor is applied to ensure
    energy preservation across scales.
    """
    # Angular frequencies with for even/odd signal lengths
    freq_factor = 2.0 * np.pi / (n * dt)  # Angular frequency factor
    
    if n % 2:  # odd sample size
        w = np.arange(1, n // 2 + 1)
        w = np.concatenate(([0], w, -w[::-1])) * freq_factor
    else:  # even sample size
        w = np.arange(1, n // 2 + 1)
        w = np.concatenate(([0], w, -w[:-1][::-1])) * freq_factor
    
    # Reshape scales to allow broadcasting with frequencies
    scales = scales[:, np.newaxis]
    
    # Compute scaled wavelets in Fourier space efficiently
    psiscaled_ft = wavelet['psi_fft'](scales * w)
    
    # Apply L2 normalization if specified
    if wavelet['norm'] == 'L2':
        psiscaled_ft *= (scales / dt)**0.5
    
    return psiscaled_ft

def compute_wavelet_coef(x: np.ndarray, 
                         psiscaled_ft: np.ndarray, 
                         chunk_size: Optional[int] = None, 
                         use_pyfftw: bool = False, 
                         max_workers: Optional[int] = None) -> np.ndarray:
    """
    Compute wavelet coefficients using the FFT convolution method with memory management.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input signal to analyze.
    psiscaled_ft : numpy.ndarray
        Scaled wavelet in Fourier space, with shape (n_scales, signal_length).
    chunk_size : int, optional (default=None)
        Size of chunks to process at once to save memory. If None,
        process all scales at once. Use this for very large scale arrays.
    use_pyfftw : bool, optional (default=False)
        Whether to use the pyfftw library if available for faster FFT computation.
        If True but pyfftw is not installed, falls back to scipy.
    max_workers : int, optional (default=None)
        Maximum number of worker processes for parallel computation of chunks.
        If None, uses the number of CPU cores - 1. Only used when chunk_size is set.
    
    Returns
    -------
    wave : numpy.ndarray
        Wavelet coefficients with shape (n_scales, signal_length).
        
    Notes
    -----
    This function computes wavelet coefficients using the Fast Fourier Transform (FFT)
    convolution method, which is more efficient than direct convolution for large signals.
    
    The implementation includes multiple optimizations:
    - Automatic memory management by reducing precision if memory errors occur
    - Optional chunked processing for large scale arrays
    - Option to use pyfftw for faster FFT computation
    - Parallel processing of chunks when appropriate
    
    The computation follows these steps:
    1. FFT of the input signal (computed once)
    2. Multiplication with scaled wavelets in Fourier space (per scale or chunk)
    3. Inverse FFT to obtain wavelet coefficients
    
    For extremely large datasets, specify a chunk_size to process scales in batches.
    """
    import scipy as sp
    
    # Optional imports for optimization
    parallel_processing = False
    fftw_available = False
    
    # Try to import pyFFTW for faster FFT if requested
    if use_pyfftw:
        try:
            import pyfftw
            from multiprocessing import cpu_count
            fftw_available = True
            # PyFFTW setup for better performance
            pyfftw.interfaces.cache.enable()
            threads = max(1, cpu_count() - 1)  # Leave one core free
        except ImportError:
            pass
    
    # Try to import concurrent.futures for parallel processing if chunking is used
    if chunk_size is not None:
        try:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import cpu_count
            parallel_processing = True
            if max_workers is None:
                max_workers = max(1, cpu_count() - 1)  # Leave one core free
        except ImportError:
            pass
    
    # Get the number of scales and signal length
    n_scales, n_signal = psiscaled_ft.shape
    
    # Compute FFT of the input signal (only once)
    try:
        # Try using FFTW if available and requested
        if fftw_available:
            y_ft = pyfftw.interfaces.scipy_fftpack.fft(x, threads=threads)
        else:
            y_ft = sp.fft.fft(x)#workers=-1
    except Exception as e:
        # Fall back to scipy's FFT
        print(f"Warning: Error during FFT computation: {e}. Using scipy fallback.")
        y_ft = sp.fft.fft(x)
    
    # Prepare output array for results
    try:
        # Try to pre-allocate the full results array
        wave = np.zeros((n_scales, n_signal), dtype=np.complex128)
    except MemoryError:
        # If memory error during allocation, try with reduced precision
        try:
            print('Reducing precision of output array to 64 bits due to memory shortage...')
            wave = np.zeros((n_scales, n_signal), dtype=np.complex64)
        except MemoryError:
            # If we can't allocate the array even with reduced precision, and
            # no chunk_size was provided, we should raise an error immediately
            if chunk_size is None:
                raise MemoryError(
                    'Not enough memory to allocate output array. '
                    'Try using chunk_size parameter to process in smaller batches.'
                )
            # For chunked processing, we'll create the array during that phase
            # but first double-check we'll be able to allocate at least a portion
            if n_scales > 0 and chunk_size > 0:
                try:
                    # Try allocating just one chunk to verify it's possible
                    test_size = min(chunk_size, n_scales)
                    test_array = np.zeros((test_size, n_signal), dtype=np.complex64)
                    del test_array  # Clean up immediately
                    print('Severe memory constraints detected. Will attempt to allocate array during chunked processing.')
                    wave = None
                except MemoryError:
                    # If we can't even allocate a single chunk, we can't proceed
                    raise MemoryError(
                        'Extreme memory constraints. Unable to allocate even a single chunk. '
                        'Try reducing chunk_size further, or processing fewer scales.'
                    )
            else:
                raise ValueError('Invalid chunk_size or n_scales. Must be positive values.')
    
    # Function to process a single chunk of scales
    def process_chunk(chunk_indices):
        chunk_start, chunk_end = chunk_indices
        chunk_psiscaled_ft = psiscaled_ft[chunk_start:chunk_end]
        
        try:
            # Try computation with full precision
            if fftw_available:
                chunk_wave = pyfftw.interfaces.scipy_fftpack.ifft(
                    y_ft * chunk_psiscaled_ft, axis=1, threads=threads)
            else:
                chunk_wave = sp.fft.ifft(y_ft * chunk_psiscaled_ft, axis=1)
        except MemoryError:
            # If memory error, reduce precision
            print(f'Reducing precision for chunk {chunk_start}-{chunk_end} due to memory shortage...')
            
            # Convert to lower precision
            chunk_psiscaled_ft_reduced = chunk_psiscaled_ft
            y_ft_reduced = y_ft
            
            if chunk_psiscaled_ft.dtype == np.complex128:
                chunk_psiscaled_ft_reduced = chunk_psiscaled_ft.astype(np.complex64)
            
            if y_ft.dtype == np.complex128:
                y_ft_reduced = y_ft.astype(np.complex64)
            
            try:
                # Try with reduced precision
                if fftw_available:
                    chunk_wave = pyfftw.interfaces.scipy_fftpack.ifft(
                        y_ft_reduced * chunk_psiscaled_ft_reduced, axis=1, threads=threads)
                else:
                    chunk_wave = sp.fft.ifft(y_ft_reduced * chunk_psiscaled_ft_reduced, axis=1)
            except MemoryError:
                # If still out of memory, provide specific guidance for this chunk
                raise MemoryError(
                    f'Not enough memory to process scales {chunk_start}-{chunk_end}. '
                    f'Try a smaller chunk_size.'
                )
        
        return chunk_start, chunk_end, chunk_wave
    
    # Process all scales at once if no chunking is requested
    if chunk_size is None:
        try:
            # Try computation with full precision
            if fftw_available:
                wave = pyfftw.interfaces.scipy_fftpack.ifft(y_ft * psiscaled_ft, axis=1, threads=threads)
            else:
                wave = sp.fft.ifft(y_ft * psiscaled_ft, axis=1)
        except MemoryError:
            # If memory error, reduce precision
            print('Reducing precision of arrays to 32 bits due to memory shortage...')
            
            # Convert to lower precision
            psiscaled_ft_reduced = psiscaled_ft
            y_ft_reduced = y_ft
            
            if psiscaled_ft.dtype == np.complex128:
                psiscaled_ft_reduced = psiscaled_ft.astype(np.complex64)
            
            if y_ft.dtype == np.complex128:
                y_ft_reduced = y_ft.astype(np.complex64)
            
            try:
                # Try computation with reduced precision
                if fftw_available:
                    wave = pyfftw.interfaces.scipy_fftpack.ifft(
                        y_ft_reduced * psiscaled_ft_reduced, axis=1, threads=threads)
                else:
                    wave = sp.fft.ifft(y_ft_reduced * psiscaled_ft_reduced, axis=1)
            except MemoryError:
                # If still out of memory, suggest chunked processing
                suggested_chunk = max(1, n_scales // 4)
                raise MemoryError(
                    'Not enough memory to compute all wavelet coefficients at once. '
                    f'Try setting chunk_size={suggested_chunk} to process in batches.'
                )
    else:
        # Process scales in chunks to save memory
        chunk_indices = [(i, min(i + chunk_size, n_scales)) 
                          for i in range(0, n_scales, chunk_size)]
        
        # Initialize output array if not already done
        if wave is None:
            # Try to create the array with a more memory-efficient approach
            try:
                # Try allocating just a small test chunk first to check memory availability
                chunk_start, chunk_end = chunk_indices[0]
                chunk_size_actual = chunk_end - chunk_start
                # Test allocation with complex128
                temp_chunk = np.zeros((chunk_size_actual, n_signal), dtype=np.complex128)
                # If successful, allocate the full array
                wave = np.zeros((n_scales, n_signal), dtype=np.complex128)
                # Clean up test allocation
                del temp_chunk
            except MemoryError:
                try:
                    # Try with reduced precision (complex64)
                    print('Reducing precision to 64 bits due to memory constraints...')
                    chunk_start, chunk_end = chunk_indices[0]
                    chunk_size_actual = chunk_end - chunk_start
                    # Test allocation with complex64
                    temp_chunk = np.zeros((chunk_size_actual, n_signal), dtype=np.complex64)
                    # If successful, allocate the full array
                    wave = np.zeros((n_scales, n_signal), dtype=np.complex64)
                    # Clean up test allocation
                    del temp_chunk
                except MemoryError:
                    # If we still can't allocate memory even with reduced precision,
                    # suggest more aggressive chunking or other strategies
                    raise MemoryError(
                        'Extreme memory constraints. Unable to allocate output array. '
                        'Try setting an even smaller chunk_size or processing fewer scales at once.'
                    )
        
        # Process chunks sequentially or in parallel
        if parallel_processing and len(chunk_indices) > 1:
            # Parallel processing of chunks
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_chunk, indices) for indices in chunk_indices]
                
                for future in futures:
                    try:
                        chunk_start, chunk_end, chunk_wave = future.result()
                        wave[chunk_start:chunk_end] = chunk_wave
                    except Exception as e:
                        print(f"Error in parallel processing: {e}")
                        # Fall back to sequential processing if parallel fails
                        for chunk_start, chunk_end in chunk_indices:
                            _, _, chunk_wave = process_chunk((chunk_start, chunk_end))
                            wave[chunk_start:chunk_end] = chunk_wave
                        break
        else:
            # Sequential processing of chunks
            for chunk_start, chunk_end in chunk_indices:
                _, _, chunk_wave = process_chunk((chunk_start, chunk_end))
                wave[chunk_start:chunk_end] = chunk_wave
    
    return wave


def define_coi(wavelet: Dict[str, Any], 
               n: int, 
               dt: Union[int, float], 
               percentout: float = 0.02, 
               fct: Optional[Callable] = None, 
               unit: str = 'p', 
               fper: Optional[ArrayLike] = None, 
               metric: str = 'int') -> np.ndarray:
    """
    Compute the Cone of Influence (COI) for wavelet transforms.
    
    The Cone of Influence identifies regions of the wavelet transform where edge effects
    become significant. This function computes COI values based on the wavelet type and
    the percentage of wavelet energy that extends beyond the signal boundaries.
    
    Parameters
    ----------
    wavelet : dict
        Output of define_wavelet function containing:
        - 'name': Wavelet name
        - 'fourier_fac': Fourier factor for the wavelet
        - 'params': Wavelet parameters
    n : int
        Signal length
    dt : int or float
        Time step between consecutive samples
    percentout : float, optional (default=0.02, 2%)
        Percentage (0-1) of wavelet energy outside signal edges considered significant.
    fct : callable, optional (default=None)
        Function to convert time units (output of data_utils.extract_duration_info).
    unit : str, optional (default='p')
        Unit for COI values: 'p' for periods, 'f' for frequencies.
    fper : array_like, optional (default=None)
        Periods or frequencies array for scaling COI values.
    meth : str, optional (default='int', 'integral')
        Method for calculating percentage: 'integral' (L1 norm) or 'energy' (L2 norm).
        
    Returns
    -------
    COI : numpy.ndarray
        Cone of Influence values for each time point
        
    Notes
    -----
    The COI calculation depends on the wavelet type:
    - For DOG wavelets: Analytical formula for DOG1, numerical integration for higher orders
    - For Haar wavelets: Simple analytical formula based on support
    - For Morlet/Gaussian wavelets: Analytical formula based on Gaussian decay
    
    The returned COI values have the same units as the input time steps (dt),
    or are converted according to the provided conversion function (fct) or
    frequency information (fs and fper).
    """
    # Extract parameters from wavelet
    wname = wavelet['name']
    fourier_fac = wavelet['fourier_fac']
    s2 = wavelet['params']['s2']
    
    # Validate unit
    if unit[0].lower() not in ['p', 'f']:
        raise ValueError("unit must be 'p' for periods or 'f' for frequencies")

    # Calculate COI value based on wavelet type
    if wname[:3] == 'dog':  # Derivative of Gaussian wavelets
        m = int(wname[3:])
        if m == 1:  # Analytical expression for DOG1
            coival = fourier_fac / np.sqrt(-2 * s2 * np.log(percentout*2))
        else:  # Numerical integration for higher-order DOG wavelets
            coival = dog_coi_factor(m, s2, fourier_fac, metric, percentout)
    elif wname == 'haa':  # Haar wavelet
        coival = fourier_fac / (0.5 - percentout)
    else:  # Morlet wavelet and Gaussian function
        coival = fourier_fac / np.sqrt(-2 * s2 * np.log(percentout))
    
    # Create distance indices from the center of the signal
    # Handle both odd and even signal lengths correctly
    if n % 2:  # Odd sample size
        half_n = int(np.ceil(n / 2))
        indices = np.arange(1, half_n + 1)
        indices = np.concatenate((indices, indices[-2::-1]))  # COI symmetric w.r.t center
    else: # even sample size
        half_n = int(n / 2)
        indices = np.arange(1, half_n + 1)
        indices = np.concatenate((indices, indices[::-1]))

    # Calculate COI values proportional to distance from edges
    COI = coival * dt * indices

    # Apply conversion function (if ts is a timedelta)
    if fct is not None:
        vectorized_fct = np.vectorize(fct)
        COI = vectorized_fct(COI)

    # Ensure COI values don't exceed maximum frequency / minimum period
    if fper is not None:
        if unit[0].lower() == 'p':
            COI[COI < np.min(fper)] = np.min(fper)
        else:
            COI = 1.0 / COI
            COI[COI > np.max(fper)] = np.max(fper)
    
    return COI

def dog_coi_factor(m: int, 
                   s2: Union[int, float], 
                   fourier_fac: float, 
                   metric: str = 'integral', 
                   percentout: float = 0.02) -> float:
    """
    Compute Cone of Influence (COI) for Derivative of Gaussian wavelets using numerical integration.
    
    The COI identifies regions of the wavelet transform where edge effects become significant.
    This function computes the COI value based on the percentage of wavelet energy or area
    that falls outside the signal boundaries.
    
    Parameters
    ----------
    m : int
        Order of derivation of the DOG function (>0)
    s2 : int or float
        Variance of the Gaussian function used to define wavelets
    fourier_fac : float
        Fourier factor, defined as the ratio of period to scale
    metric : str, optional (default='integral')
        Metric to define qcoi, either 'integral' (area) or 'energy' (squared area).
        Default is 'integral'.
    percentout : float, optional (default=0.02)
        Percentage (0-1) of wavelet energy or area outside signal edges.
        
    Returns
    -------
    coival : float
        Computed COI value that converts scales to time units
        
    Notes
    -----
    The function works by finding the point where the integral of the wavelet function
    (or its square) reaches the threshold percentage (percentout) of the total integral.
    
    For Derivative of Gaussian wavelets, this involves analyzing the roots of the
    corresponding Hermite polynomials and performing numerical integration between them.
    
    The implementation includes two metrics:
    - 'integral': Based on the L1 norm (integral of absolute values)
    - 'energy': Based on the L2 norm (integral of squared values)
    
    Raises
    ------
    ValueError
        If metric is not 'integral' or 'energy', or if percentout is not in range (0,1)
    """
    import sympy as smp
    from scipy.optimize import brentq
    
    # Method check and validation
    if str(metric).lower().startswith(("l1", "int")):
        metric = 'integral'  # Compute COI based on int(-Inf,Inf)|f(x)|dx
    elif str(metric).lower().startswith(("l2", "ener")):
        metric = 'energy'  # Compute COI based on int(-Inf,Inf)|f(x)|^2dx
    else:
        raise ValueError(f'Method for area computation "{metric}" inadmissible. '
                         'Available methods are "integral" or "energy"')
    
    # Percentage threshold validation
    if not isinstance(percentout, float) or not (0 < percentout < 1):
        raise ValueError(f"Percentage {percentout} inadmissible. "
                         "Must be a float in the range (0, 1)")
    
    # Create symbolic variable for mathematical expressions
    xvar = smp.symbols('xvar')
    
    # Define the DOG wavelet function symbolically
    # m-th derivative of Gaussian
    expr = smp.diff(smp.exp(-xvar**2 / (2 * s2)), xvar, m)

    # Calculate Hermite polynomial roots which correspond to DOG wavelet zero-crossings
    hermite_poly = smp.hermite(m, xvar / smp.sqrt(2 * s2))
    sols = sorted(smp.nroots(hermite_poly))
    # Convert symbolic solutions to floating-point numbers
    sols = [float(sol) for sol in sols]
    
    # Initialize algorithm variables
    value = 0  # Current integral value
    Limit_Inf = -np.inf  # Lower integration bound
    Limit_Sup = sols[0]  # Upper integration bound (first root)
    k = 0  # Root index counter
    
    # Implementation varies by method (integral vs. energy)
    if metric == 'integral':
        # Get total integral value over entire domain
        int_value = Retrieve_DOG_integral(m, s2)
        
        # Target threshold value (percentage of total integral)
        target = percentout * int_value
        
        # For integral method, use analytically determined antiderivative
        # This is (m-1)th derivative of Gaussian
        int_expr = smp.diff(smp.exp(-xvar**2 / (2 * s2)), xvar, m-1)
        
        # Iterate through roots to find the right interval
        # This breaks the integral into segments between consecutive roots
        addv = float(int_expr.subs(xvar, Limit_Sup))
        while value + addv < target and k < len(sols) - 1:
            k += 1
            value += addv
            Limit_Inf = Limit_Sup
            Limit_Sup = sols[k]
            # Calculate difference in antiderivative between roots
            addv = abs(float(int_expr.subs(xvar, Limit_Sup) - int_expr.subs(xvar, Limit_Inf)))
        
        # Handle edge case when starting from -infinity
        if Limit_Inf == -np.inf:
            # Set a practical finite lower bound based on Gaussian decay
            Limit_Inf = (-3 - m/10) * s2**(1/2)
            value = float(int_expr.subs(xvar, Limit_Inf))
            
        # Convert symbolic expression to numerical function
        int_expr_func = smp.lambdify(xvar, int_expr, 'numpy')
        LB = int_expr_func(Limit_Inf)
        
        # Function to find the precise point where integral equals target
        def integral_difference(upper_limit):
            integrated_expr = abs(int_expr_func(upper_limit) - LB)
            return integrated_expr - (target - value)
        
    elif metric == 'energy':
        # For energy method, use numerical integration of squared function
        from scipy.special import gamma
        from scipy.integrate import quad, IntegrationWarning
        import warnings
        # Suppress integration warnings for more readable output
        warnings.filterwarnings('ignore', category=IntegrationWarning)
        
        # Total energy (L2 norm) - analytical formula for Gaussian derivatives
        int_value = gamma(m + 0.5) / (s2**(m - 0.5))
        
        # Target threshold value
        target = percentout * int_value
        
        # Square the expression for energy calculation
        expr_squared = expr**2
        # Convert to numerical function for integration
        func = smp.lambdify(xvar, smp.Abs(expr_squared), 'numpy')
        
        # Numerical integration between consecutive roots
        addv, _ = quad(func, Limit_Inf, Limit_Sup, epsabs=1e-15, epsrel=1e-15, limit=1000)
        while value + addv < target and k < len(sols) - 1:
            k += 1
            value += addv
            Limit_Inf = Limit_Sup
            Limit_Sup = sols[k]
            addv, _ = quad(func, Limit_Inf, Limit_Sup, epsabs=1e-15, epsrel=1e-15, limit=1000)

        # Handle edge case when starting from -infinity
        if Limit_Inf == -np.inf:
            Limit_Inf = (-3 - m/10) * s2**(1/2)
            value, _ = quad(func, -np.inf, Limit_Inf, epsabs=1e-15, epsrel=1e-15, limit=1000)
        
        # Function to find the precise point where integral equals target
        def integral_difference(upper_limit):
            integrated_expr, _ = quad(func, Limit_Inf, upper_limit)
            return integrated_expr - (target - value)
    
    # Use Brent's method to find the root numerically within the interval
    # This finds the exact point where the integral reaches the target value
    root = brentq(integral_difference, Limit_Inf, Limit_Sup)
    
    # Convert to period (coival = period/b)
    # The negative sign is because root = -b/a in the COI formulation
    coival = -fourier_fac / root

    return coival

def define_coi_dirac(wavelet: Dict[str, Any], 
                     n: int, 
                     ts: Union[int, float, "timedelta"], 
                     scales: np.ndarray, 
                     threshold: float = 0.02, 
                     metric: str = 'integral') -> np.ndarray:
    """
    Compute Cone of Influence (COI) using Dirac impulses at the edges of the signal.
    
    This function directly computes the COI by analyzing the wavelet transform of Dirac
    impulses placed at the edges of the signal. This provides a data-driven approach
    to determine edge effects without relying on analytical formulas.
    
    Parameters
    ----------
    wavelet : dict
        Output of define_wavelet function containing:
        - 'psi_fft': The wavelet function in Fourier space
        - 'norm': Normalization method ('L1' or 'L2')
        - 'fourier_fac': Fourier factor for the wavelet
    n : int
        Signal length
    ts : int, float or datetime.timedelta
        Sampling period
    scales : numpy.ndarray
        Scales used for the wavelet transform
    threshold : float, optional (default=0.02, 2%)
        Percentage of edge effect considered significant
    metric : str, optional (default='integral')
        Metric to compute threshold, based on absolute wavelet coefficients ('integral') 
        or squared wavelet coefficients ('energy').
        
    Returns
    -------
    coi : numpy.ndarray
        COI values for each time point
        
    Notes
    -----
    The algorithm works by:
    1. Creating a signal with Dirac impulses at the edges
    2. Computing its wavelet transform
    3. Measuring how the edge effects decay with distance from the edge
    4. Finding the contour where edge effects drop below the threshold
    
    This approach can be more accurate than analytical formulas for complex wavelets
    or when exact edge effect behavior is important.
    """
    from scipy.interpolate import interp1d
    from datetime import timedelta
    
    # Create a test signal with Dirac impulses at the edges
    # We use length n+2 to ensure the impulses are at the true edges
    dirac = np.zeros(n + 2)
    dirac[[0, -1]] = 1  # Set impulses at first and last positions

    # Extract time information and handle datetime objects
    dt = ts
    fct = None
    # If ts is a timedelta, extract numerical value and conversion function
    if isinstance(ts, timedelta):
        try:
            import data_utils   # Dynamically import to avoid dependency issues
            
            dt, fct, _ = data_utils.extract_duration_info(ts)
        except ImportError:
            raise ImportError("data_utils module is required for timedelta processing")
        except Exception as e:
            raise ValueError(f"Error processing timedelta objects: {str(e)}")
    
    # Compute the wavelet transform of the Dirac impulses
    # First compute the wavelet function at each scale
    psi_scales = compute_psi(n + 2, dt, wavelet['psi_fft'], scales, wavelet['norm'])
    
    # Then compute the wavelet transform
    wave = compute_wavelet_coef(dirac, psi_scales)
    
    # Calculate the power (squared amplitude) of the transform
    power = np.abs(wave) if metric.lower()[:3] == 'int' else np.abs(wave)**2
    
    # Normalize the power to get relative edge effects
    power_max = np.max(power, axis=1, keepdims=True)
    norm_power = power / power_max
    
    # Find the contour at the threshold value (e.g., 0.02 or 2%)
    # For each scale, find where edge effects drop below threshold
    coi = np.zeros(n)
    
    # Convert scales to periods for output
    periods = wavelet['fourier_fac'] * scales
    
    # Determine the COI at each time point
    # First half from the left edge, second half from the right edge
    for i, scale_idx in enumerate(range(len(scales))):
        # Extract normalized power for this scale
        scale_power = norm_power[scale_idx]
        
        # Find where power drops below threshold from left edge
        left_below = np.where(scale_power[:n//2+1] < threshold)[0]
        if len(left_below) > 0:
            left_idx = left_below[0]
        else:
            left_idx = n//2  # Default if threshold never reached
        
        # Find where power drops below threshold from right edge
        right_below = np.where(scale_power[n//2+1:] < threshold)[0]
        if len(right_below) > 0:
            right_idx = n//2 + 1 + right_below[-1]
        else:
            right_idx = n//2 + 1  # Default if threshold never reached
        
        # Set the period value at the corresponding indices
        if i < len(periods):
            if left_idx < n:
                coi[left_idx] = max(coi[left_idx], periods[i])
            if right_idx < n:
                coi[right_idx] = max(coi[right_idx], periods[i])
    
    # Fill in missing values with interpolation
    valid_indices = np.where(coi > 0)[0]
    if len(valid_indices) > 1:  # Need at least 2 points to interpolate
        interpolator = interp1d(
            valid_indices, 
            coi[valid_indices],
            bounds_error=False,
            fill_value=(coi[valid_indices[0]], coi[valid_indices[-1]])
        )
        
        # Generate interpolated COI for all indices
        all_indices = np.arange(n)
        coi = interpolator(all_indices)
    
    # Apply units conversion if needed
    if fct is not None:
        vectorized_fct = np.vectorize(fct)
        coi = vectorized_fct(coi)
    
    return coi



def compute_significance(y: np.ndarray, 
                         wavelet: Dict[str, Any], 
                         dt: Union[int, float], 
                         wave: np.ndarray, 
                         periods: np.ndarray, 
                         lag1: Optional[float] = None, 
                         prob: float = 0.95) -> np.ndarray:
    """
    Compute the significance levels for wavelet coefficients.
    
    This function determines which wavelet coefficients are statistically significant
    compared to a background red noise process (AR1 process).
    
    Parameters
    ----------
    y : numpy.ndarray
        Original time series data
    wavelet : dict
        Dictionary containing wavelet parameters
    dt : int or float
        Time step between consecutive samples
    wave : numpy.ndarray
        Wavelet coefficients array
    scales : numpy.ndarray
        Scales used for the wavelet transform
    periods : numpy.ndarray
        Periods corresponding to the scales
    lag1 : float, optional (default=None)
        Lag-1 autocorrelation coefficient. If None, it will be estimated from the data.
    prob : float, optional (default=0.95, 95% confidence level)
        Significance level (0-1).
        
    Returns
    -------
    signif : numpy.ndarray or None
        Boolean array indicating which coefficients are significant (True) or not (False).
        Returns None if lag1 is None.
        
    Notes
    -----
    The significance testing follows the method described in Torrence and Compo (1998):
    "A Practical Guide to Wavelet Analysis" Bull. Amer. Meteor. Soc., 79, 61-78.
    
    The function uses an AR1 model to estimate the power spectrum of the background (assumed) red noise process.
    
    This implementation accounts for normalization differences between L1 and L2 norms
    when computing significance levels.
    """
    # Calculate significance thresholds for each period
    signif_values = compute_significance_threshold(periods, dt, wavelet, y, lag1, prob)
    # If no lag1 provided or computed, we can't determine significance
    if signif_values is None:
        return None
    # Replicate the significance values for each time point
    signif_values = signif_values[:, np.newaxis] 
    
    # Adjust power values based on normalization method
    if wavelet['norm'] == 'L1':  
        # For L1 norm, we need to convert power to match theoretical spectrum
        # Compute scale-dependent factor
        scales = periods / wavelet['fourier_fac']
        scale_factors = (scales / dt) # **0.5 removed since squared taken after
        # Compute power after renormalization to maintain L2 energy conservation
        powersig = np.abs(wave)**2 * (wavelet['renorm_fac']**2) * scale_factors[:, np.newaxis] # squared removed

        # Compare with significance threshold
        signif_ratio = powersig / signif_values
    else:
        # For L2 norm, direct comparison is valid
        signif_ratio = np.abs(wave)**2 / signif_values
    
    # Return boolean array where True indicates significance
    return signif_ratio >= 1

def compute_significance_threshold(periods: np.ndarray, 
                                  dt: Union[int, float], 
                                  wavelet: Dict[str, Any], 
                                  signal: np.ndarray, 
                                  lag1: Optional[float] = None, 
                                  prob: float = 0.95) -> np.ndarray:
    """
    Compute significance threshold of wavelet coefficients for each scale
    assuming a red background noise (AR1 process).
    
    Parameters
    ----------
    periods : numpy.ndarray
        Array of periods at which to calculate significance thresholds
    dt : float
        Time step between consecutive samples
    wavelet : dict
        Dictionary containing wavelet parameters including 'is_complex'
        to determine degrees of freedom
    signal : numpy.ndarray
        Original time series data used to estimate lag1 autocorrelation
        and variance if not provided
    prob : float, optional (default=0.95, 95% confidence level)
        Probability level for significance (0-1).
    lag1 : float, optional (default=None)
        Lag-1 autocorrelation coefficient. If None, will be estimated from signal
        
    Returns
    -------
    wavsig : numpy.ndarray or None
        Array of significance threshold values corresponding to each period.
        Returns None if lag1 estimation fails.
        
    Notes
    -----
    The theoretical background is based on Torrence & Compo (1998): 
    "A Practical Guide to Wavelet Analysis" Bull. Amer. Meteor. Soc., 79, 61-78.
    
    The function calculates the theoretical red noise spectrum based on lag1 
    autocorrelation and signal variance, then determines threshold values using 
    chi-square distribution with appropriate degrees of freedom.
    
    """
    def ar1nv(x: Union[ArrayLike, "pd.Series", "pd.DataFrame"]) -> float:
        """
        Estimate the lag-1 autocorrelation coefficient for an AR(1) model.
        
        Parameters
        ----------
        x : array_like
            Input time series data
            
        Returns
        -------
        g : float
            Estimate of the lag-1 autocorrelation coefficient
            
        Notes
        -----
        This function computes the lag-1 autocorrelation for an AR(1) process
        using the formula: r(1) = c(1)/c(0) where c(k) is the autocovariance
        at lag k.
        
        The original implementation by Eric Breitenberger also included the
        calculation of noise variance, which is currently commented out.
        
        For time series with significant trends or non-stationarity, the
        estimation may be inaccurate and detrending might be necessary
        before calling this function.
        """
        # Convert input to numpy array and flatten
        x = np.asarray(x).flatten()
        N = len(x)
        
        # Remove mean (detrend by constant)
        m = np.mean(x)
        x = x - m

        # Compute lag-0 (variance) and lag-1 covariance estimates
        c0 = np.dot(x, x) / N  # Lag-0 autocovariance (variance)
        c1 = np.dot(x[:-1], x[1:]) / (N - 1)  # Lag-1 autocovariance

        # Lag-1 autocorrelation coefficient
        g = c1 / c0
        
        # Option to compute noise variance (commented out)
        # a = np.sqrt((1 - g**2) * c0)

        return g

    from scipy.stats import chi2
    
    # Convert periods to normalized frequencies (0-0.5)
    norm_freq = dt / periods

    # Set degrees of freedom based on wavelet type
    if wavelet.get('is_complex', False):
        dof = 2
    else:
        dof = 1

    # Estimate AR1 coefficient if not provided
    if lag1 is None:
        try:
            # Attempt to estimate lag1 coefficient from the signal
            lag1 = ar1nv(signal)
            if np.isnan(lag1):
                import warnings
                warnings.warn("AR1 autocorrelation estimation failed for significance computation."
                            "\nPlease provide lag1 manually when calling the function.", UserWarning, stacklevel=2)
                # If estimation fails, return None
                return None
        except Exception as e:
            import warnings
            warnings.warn(f"Error estimating lag1 coefficient: {e}"
                        "\nSpecify lag1 manually or check your signal data.", UserWarning, stacklevel=2)
            # If estimation fails, return None
            return None
    
    # Calculate signal variance for power spectrum scaling
    variance = np.var(signal)

    # Theoretical power spectrum of AR1 process
    # P(f) = (1-lag1²) / (1+lag1² - 2*lag1*cos(2πf))
    # Scaled by signal variance to match actual data power
    P = (1 - lag1**2) / (1 + lag1**2 - 2 * lag1 * np.cos(2 * np.pi * norm_freq))
    P *= variance
    
    # Chi-square value for desired probability level
    X2 = chi2.ppf(prob, dof)
    
    # Final significance threshold: background spectrum * chi-square / dof
    wavsig = P * X2 / dof

    return wavsig




def plot_scalogram(wave: np.ndarray, 
                   fper: np.ndarray, 
                   ts: Optional[Union[int, float, "timedelta"]] = 1, 
                   xdata: Optional[Union[np.ndarray, "pd.Series", "pd.DatetimeIndex", "DatetimeArray"]] = None, 
                   coi: Optional[np.ndarray] = None, 
                   fper_units: str = 'period', 
                   signif: Optional[np.ndarray] = None, 
                   plotunits: Optional[str] = None, 
                   cmap: str = 'viridis', 
                   title: str = 'Scalogram', 
                   return_fig: bool = False) -> Tuple[Optional["matplotlib.figure.Figure"], Optional["matplotlib.axes.Axes"]]: 
    """
    Create and display a wavelet scalogram plot.
    
    This function visualizes wavelet transform coefficients as a scalogram,
    with period/frequency on the y-axis and time on the x-axis. It also
    supports plotting the cone of influence (COI) and significance contours.
    
    Parameters
    ----------
    wave : numpy.ndarray
        Wavelet coefficients matrix with shape (n_scales, n_times)
    fper : numpy.ndarray
        Periods or frequencies corresponding to each scale row in wave
    ts : int, float or timedelta, optional (default=1)
        Time step between consecutive samples.
    xdata : numpy.ndarray or pandas.Series, pandas.DatetimeIndex or pandas.arrays.DatetimeArray, optional (default=None)
        Vector of x-axis values.
        If None, it will be generated from ts and wave shape.
        Can be numeric or datetime values.
    coi : numpy.ndarray, optional (default=None)
        Cone of influence values (must have same units as fper).
    fper_units : str, optional (default='period')
        Specify whether fper contains 'period' or 'frequency' values.
    signif : numpy.ndarray, optional (default=None)
        Boolean array indicating statistically significant coefficients. 
        Must have same shape as wave.
    plotunits : str, optional (default=None)
        Display units for periods/frequencies: 'sec', 'min', 'hour', 'day', 'year'.
        Only used for timedelta objects.
    cmap : str, optional (default='viridis')
        Matplotlib colormap name to use for the scalogram.
    title : str, optional (default='Scalogram')
        Title for the plot.
    return_fig : bool, optional (default=False)
        If True, returns the figure and axis objects instead of displaying the plot.
        
    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes, optional
        Figure and axis objects, only if return_fig is True.
        
    Notes
    -----
    This function performs appropriate handling of different time formats:
    - Numeric values: displayed directly
    - datetime objects: formatted using matplotlib's date functionality
    - timedelta objects: converted to specified units if plotunits is provided
    
    The cone of influence (COI) indicates regions where edge effects become significant.
    It is plotted as a white dashed line, with the area outside the cone shaded gray.
    
    Significance contours outline regions where wavelet coefficients are statistically
    significant compared to a background noise process.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    from pandas import Timestamp
    
    # Input validation
    if wave.ndim != 2:
        raise ValueError(f"wave must be a 2D array, got shape {wave.shape}")
        
    if len(fper) != wave.shape[0]:
        raise ValueError(f"Length of fper ({len(fper)}) must match first dimension of wave ({wave.shape[0]})")
    
    if coi is not None and len(coi) != wave.shape[1]:
        raise ValueError(f"Length of coi ({len(coi)}) must match second dimension of wave ({wave.shape[1]})")
    
    if signif is not None and signif.shape != wave.shape:
        raise ValueError(f"Shape of signif {signif.shape} must match wave {wave.shape}")
    
    # Sampling period/frequency, Periods and COI units and conversion
    Units = None
    flag_timedelta = isinstance(ts, timedelta)
    
    # Handle timedelta objects
    if flag_timedelta:
        # Check format consistency
        if coi is not None:
            if not all(isinstance(c, timedelta) for c in coi):
                raise ValueError('COI and ts must have matching formats (both timedelta)')
        if not all(isinstance(p, timedelta) for p in fper):
            raise ValueError('fper and ts must have matching formats (both timedelta)')
            
        try:
            import data_utils  # Dynamically import to avoid dependency issues
            
            # Units and dimensionless sampling period
            if plotunits is not None:  # Unit conversion needed
                fct_inv, Units = data_utils.extract_value_duration(plotunits)
                ts = fct_inv(ts)
            else:
                ts, _, Units = data_utils.extract_duration_info(ts)
                fct_inv, _ = data_utils.extract_value_duration(Units)
                
            # Dimensionless periods/frequencies
            fct_vectorized = np.vectorize(fct_inv)
            fper = fct_vectorized(fper)
            
            # Dimensionless COI
            if coi is not None:
                coi = fct_vectorized(coi)
        except ImportError:
            raise ImportError("data_utils module is required for timedelta processing")
        except Exception as e:
            raise ValueError(f"Error processing timedelta objects: {str(e)}")
    elif plotunits is not None:
        try:
            import data_utils
            _, Units = data_utils.extract_value_duration(plotunits)
        except ImportError:
            raise ImportError("data_utils module is required when using plotunits")

    # Time vector
    if xdata is None:
        xdata = np.arange(0, wave.shape[1] * ts, ts)
        X = 0  # Numeric time type
    elif isinstance(xdata[0], (np.datetime64, datetime, Timestamp)):
        xdata = mdates.date2num(xdata)
        X = 1  # Datetime type
    else:
        X = 2  # Other numeric type
    
    if len(xdata) != wave.shape[1]:
        raise ValueError(f'Time vector length ({len(xdata)}) does not match signal dimension ({wave.shape[1]})')
   
    # Set y-axis properties based on period/frequency
    if fper_units.lower().startswith('p'):
        Ypers = [min(fper), max(fper)]
        ylbl = 'Period'
    elif fper_units.lower().startswith('f'):
        Ypers = [max(fper), min(fper)]
        ylbl = 'Frequency'
    else:
        raise ValueError(f'fper_units value "{fper_units}" is invalid. Please use "period" or "frequency"')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot scalogram using pcolormesh for better performance
    mesh = ax.pcolormesh(xdata, fper, np.abs(wave), shading='auto', cmap=cmap)
    
    # Configure axes
    ax.set_yscale('log')
    ax.set_ylim(Ypers)
    ax.set_xlim([min(xdata), max(xdata)])
    ax.set_title(title)
    
    # Add units to y-label if available
    if Units is not None:
        ylbl += f" ({Units})"
    ax.set_ylabel(ylbl)
    
    # Configure x-axis
    if X == 0:
        if Units is not None:
            ax.set_xlabel(f'Time ({Units})')
        else:
            ax.set_xlabel('Time')
    elif X == 1:
        ax.xaxis_date()
        fig.autofmt_xdate()
        ax.set_xlabel('Date')
    else:
        ax.set_xlabel('Time')
    
    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Amplitude')
    
    # Plot cone of influence if available
    if coi is not None:
        ax.plot(xdata, coi, 'w--', linewidth=2, label='Cone of Influence')
        ax.fill_between(xdata, coi, Ypers[1], facecolor='gray', alpha=0.6)
    
    # Plot significance contours if available
    if signif is not None:
        if np.any(signif):
            # Convert boolean array to float (1 for True, 0 for False)
            signif_float = signif.astype(float)
            
            # Plot contours
            ax.contour(xdata, fper, signif_float, colors='k', linewidths=0.5, levels=[0.5])
    
    # # Add grid for better readability
    # ax.grid(True, alpha=0.3)
    
    # Return figure or show plot
    if return_fig:
        return fig, ax
    else:
        plt.tight_layout()
        plt.show()