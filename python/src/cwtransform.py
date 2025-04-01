"""
Created on Fri Feb 14 2025

@author: Jonathan Bitton
"""
# Standard library imports
from typing import Dict, List, Optional, Sequence, Union, TYPE_CHECKING

# Third-party imports
import numpy as np

# Local imports
import wt_utils
import data_utils
import wai_utils

# Variable type hints
if TYPE_CHECKING:
    import pandas as pd
    from pandas.arrays import DatetimeArray
    from numpy.typing import ArrayLike
    from datetime import timedelta

class Wavelet_Transform():
    '''
    Continuous Wavelet Transform (CWT) implementation for signal analysis.
    
    This class implements the continuous wavelet transform for 1D series data,
    allowing visualization and analysis in the position/time-frequency domain.
    
    Signal Parameters
    ----------
    y : np.ndarray, pandas.Series, pandas.DataFrame, list, tuple, optional
        Array containing the series to be analyzed. If provided at initialization, the wavelet 
        transform will be computed immediately. If y is n-dimensional, the first two signals 
        are used for cross wavelet transform. If y is a DataFrame/Series, time information is extracted
        from relevant columns ('time' or 'datetime') when available, or from the index (if datetime index).
        See function 'data_utils.extract_signal_and_time' for details.
    
    dt : float, int, optional (default=1)
        Time step between consecutive samples. Automatically calculated
        if ts or fs are provided.
        
    ts : float, int, timedelta, optional (default=1)
        Sampling period (alternative to specifying dt or fs).
        Can be a timedelta object for real time data.
        
    fs : float, int, optional (default=1)
        Sampling frequency in Hz (alternative to specifying dt or ts).

    xdata : array_like, optional (default=None)
        X-axis for the signal. If provided, this will be used for plotting.
        For timedelta objects ts, must be an array of datetime objects.
        
    Wavelet Parameters
    -----------------
    wname : str, optional (default='mor')
        Wavelet type to use. Options include:
        - 'mor' or 'morlet': Morlet wavelet (complex)
        - 'mex' or 'mexhat': Mexican hat wavelet (DOG with m=2) 
        - 'dog{m}': Derivative of Gaussian of order m
        - 'gau' or 'gauss': Gaussian function (not a true wavelet)
        - 'haa' or 'haar': Haar wavelet

        Additional formats: 'mor{w0}-{s2}', 'mex{s2}', 'dog{m}-{s2}', 'gau{s2}' sets wavelet parameters. 
        s2 and w0 are ignored in this case.
        
    norm : str, optional (default='L1' or 'L2' depending on wavelet)
        Normalization method for the wavelets:
        - 'L1': Integral of absolute value of wavelet equals 2
        - 'L2': Integral of squared wavelet equals 1

        Default is 'L2' for the Gaussian function, 'L1' for other wavelets.
        
    s2 : float, int, optional (default=1)
        Shape parameter controlling the variance of the (enveloping) Gaussian function.
        Larger values provide better frequency resolution but worse time resolution.
        
    w0 : float, int, optional (default=6)
        Central angular frequency parameter for the Morlet wavelet.
        Values below 6 may result in non-admissible wavelets.
                
    Scale Parameters
    ---------------
    scales : array_like, optional (default=None)
        Direct specification of scales to use. If provided, other scale
        parameters are ignored.
        
    dj : float, int, optional (default=0.05)
        Scale resolution for logarithmic scale spacing.
        Smaller values give finer resolution.
        
    smin, s0 : float, int, optional (default=None, calculated from permin)
        Smallest wavelet scale to analyze. Alternative to permin and fmax.
        
    smax : float, int, optional (default=None, calculated from permax)
        Largest wavelet scale to analyze. Alternative to permax and fmin.
        
    permin, pmin : float, int, optional (default=2*dt)
        Smallest period to analyze. Alternative to smin and fmax.
        
    permax, pmax : float, int, optional (default=L*dt)
        Largest period to analyze. Alternative to smax and fmin.
        
    fmin : float, int, optional (default=1/dt)
        Lowest frequency to analyze. Alternative to permax and smax.
        
    fmax : float, int, optional (default=1/(L*dt))
        Highest frequency to analyze. Alternative to permin and smin.
        
    scaletype : str, optional (default='log')
        Type of scale spacing:
        - 'log': Logarithmic spacing (default)
        - 'lin': Linear spacing
        - 'bot': Mixed spacing (log up to scut/pcut/fcut, then linear)
        
    dj_lin : float, int, optional (default=dj)
        Scale step for linear scales when scaletype='bot'.
        Defaults to the same value as dj if not specified.
        
    scut : float, int, optional (default=None)
        Transition point from logarithmic to linear scale spacing
        when scaletype='bot'. Alternative to pcut and fcut.

    pcut : float, int, optional (default=None)
        Transition period. Alternative to scut and fcut.

    fcut : float, int, optional (default=None)
        Transition frequency. Alternative to scut and pcut.
        
    Padding Parameters
    -----------------
    pad : str, optional (default='sym')
        Signal padding method:
        - 'zpd': Zero padding
        - 'sym': Symmetric padding (mirroring)
        - 'ref': Reflection padding (symmetric excluding boundaries)
        - 'per': Periodic padding
        - 'non': No padding
        
    padmode : str, optional (default='b')
        Which side(s) of the signal to pad:
        - 'l': Left side only
        - 'r': Right side only
        - 'b': Both sides
        
    Cone of influence Parameters
    ---------------------------
    coi : bool, optional (default=True)
        Whether to compute the cone of influence (region where edge
        effects are significant).
        
    coimethod : str, optional (default='ana')
        Method for computing the cone of influence:
        - 'ana' or 'analytic': Theoretical formulation
        - 'num' or 'numeric': Numerical computation with impulses
        
    coimetric : str, optional (default='int')
        Metric for calculating the COI boundary:
        - 'int' or 'integral': Based on absolute value of wavelet
        - 'ene' or 'energy': Based on squared wavelet
        
    percentout : float, optional (default=0.02)
        Threshold for COI computation, defined as the percentage of 
        wavelet area/energy outside signal (0-1 range).
        
    Significance Parameters
    -----------------
    signif : bool, optional (default=False)
        Whether to compute significance levels for the wavelet power.

    Scalogram Parameters
    -----------------
    figure : bool, optional (default=False)
        Whether to generate a scalogram plot automatically.
        
    plotunits : str, optional (default=None)
        Units for the time axis in plots (e.g., 'sec', 'min', 'hour', 'day').

    Processing Parameters
    --------------------
    use_pyfftw : bool, optional (default=False)
        Whether to use pyfftw for FFT.
    parallel_proc : bool, optional (default=False)
        Whether to use parallel processing for computing wavelet coefficients.
    workers : int, optional (default=None)
        Number of workers for parallel processing. If None, uses the number of CPU cores - 1.
    chunk_size : int, optional (default=None)
        Chunk size for parallel processing.
        
    Attributes
    ----------
    wave : numpy.ndarray
        Complex wavelet coefficients
        
    scales : numpy.ndarray
        Scales used for the transform
        
    fper : numpy.ndarray
        Periods or frequencies corresponding to each scale
        
    periods : numpy.ndarray
        Unitless copy of periods
        
    coi : numpy.ndarray, optional
        Cone of influence (if computed)
        
    signif : dict, optional
        Significance levels (if computed)
        
    Notes
    -----
    - For performance reasons, consider pre-defining scales when processing
      multiple signals with the same parameters
    '''
    
    def __init__(self, 
                 y: Optional[Union["ArrayLike", "pd.Series", "pd.DataFrame"]] = None, 
                 **args):
        """
        Initialize the Wavelet_Transform object.
        
        Parameters
        ----------
        y : array_like, optional
            Array containing the series to be analyzed. If provided,
            the wavelet transform will be computed immediately.
        **args : dict
            Keyword arguments for configuring the wavelet transform.
            See class docstring for full parameter details.
        """
        # Import and adjust parameters of the transform
        self.import_arguments(**args)

        # Define wavelet
        self.set_wavelet(**self.wav_params)
        del self.wav_params

        # Scales and periods/frequencies
        if hasattr(self, 'scales'):
            self.convert_scales()

        # Run cwt if signal provided    
        if y is not None:
            xdata = self.xdata
            self.cwt(y, xdata=xdata)
        
        del self.xdata

    def cwt(self, 
            y: Union["ArrayLike", "pd.Series", "pd.DataFrame"], 
            xdata: Optional[Union["ArrayLike", "pd.Series", "pd.DatetimeIndex", "DatetimeArray"]] = None):
        """
        Compute the Continuous Wavelet Transform of a signal.
        
        Parameters
        ----------
        y : array_like
            Input signal to analyze. If y contains two signals, the second
            signal is used for cross-wavelet transform computation.
        xdata : array_like, pandas.Series, pandas.DatetimeIndex, pandas.arrays.DatetimeArray, optional (default=None)
            X-axis data for plotting. If xdata is missing and y is a DataFrame/Series, time information is extracted
            from relevant columns ('time' or 'datetime') when available, or from the index (if datetime index).
        
            
        Returns
        -------
        None
            Results are stored as attributes of the class instance.
            
        Notes
        -----
        This method performs these steps in sequence:
        1. Extract signal and time data
        2. Apply padding to the signal
        3. Compute scales if not already defined
        4. Compute wavelet coefficients
        5. Remove padding from results
        6. Compute cone of influence (if requested)
        7. Calculate significance levels (if requested)
        8. Generate scalogram plot (if requested)
        9. Compute cross-wavelet transform (if second signal provided)
        """
        # Extract signal and xdata
        y, compute_xwt, y2, xdata = data_utils.extract_signal_and_time(y, self.flags['figure'], xdata)   
         
        # Signal length and padding
        y_ext = self.signal_and_padding(y)

        # Scales and periods/frequencies
        if hasattr(self, 'scale_param'):
            self.compute_scales(**self.scale_param)
            del self.scale_param

        # Compute wavelet 
        self.compute_WT(y_ext)

        # Remove padding
        if 'remove_padding' in self.params:
            self.wave = self.params['remove_padding'](self.wave)

        # Cone of influence
        if self.flags['coi']: 
            self.compute_coi(self.params['coimethod'], self.params['coimetric'])
        else:
            self.coi = None

        # Significance
        if self.flags['signif']: 
            self.compute_signif(y)

        # Scalogram
        if self.flags['figure']: 
            self.scalogram(xdata, self.params['plotunits'])

        # Cross wavelet transform
        if compute_xwt: 
            self.xwt(y2)
        
        
    def import_arguments(self, **args):
        '''
        Importing arguments and setting parameters
        
        Parameters
        ----------
        **args : dict
            Keyword arguments for configuring the wavelet transform.
            See class docstring for full parameter details.
            
        Notes
        -----
        This method imports the arguments and sets the parameters for the wavelet transform.
        It uses the `get_args` function from the `data_utils` module to extract the parameters.
        '''
        self.params, self.wav_params, scale_param, self.flags, self.processing, self.xdata = data_utils.get_args(**args)
        
        if isinstance(scale_param, dict):
            self.scale_param = scale_param
        else:
            self.scales = scale_param

        
    def set_wavelet(self, 
               wname: str, 
               norm: Optional[str] = None, 
               s2: Union[int, float] = 1, 
               w0: Union[int, float] = 6):
        '''
        Defining wavelet dictionary according to name and parameters
        
        Parameters
        ----------
        wname : str
            Name of the wavelet to use. See class docstring for available wavelets.
        norm : str, optional (default='L1')
            Normalization type: 'L1' or 'L2'.
        s2 : int or float, optional (default=1)
            Variance parameter for Gaussian-based wavelets.
        w0 : int or float, optional (default=6)
            Central angular frequency parameter for Morlet wavelet.
        
        Returns
        -------
        dict
            Dictionary containing wavelet properties.
        
        Notes
        -----
        This method defines the wavelet dictionary based on the specified wavelet name and parameters.
        It uses the `define_wavelet` function from the `wt_utils` module to create the wavelet dictionary.
        '''
        # Delete scaled fft wavelet if already defined
        if hasattr(self, 'psi'):
            del self.psi
        
        # Define wavelet
        self.wavelet = wt_utils.define_wavelet(wname, norm, s2, w0)
    
    def compute_scales(self, 
                       smin: Optional[Union[int, float]] = None, 
                       permin: Optional[Union[int, float]] = None, 
                       smax: Optional[Union[int, float]] = None, 
                       permax: Optional[Union[int, float]] = None, 
                       scut: Optional[Union[int, float]] = None, 
                       pcut: Optional[Union[int, float]] = None):
        '''
        Defining scales and converting to periods/frequencies
        
        Parameters
        ----------
        smin : float, optional (default=None)
            Minimum scale value.
        permin : float, optional (default=None)
            Minimum period value.
        smax : float, optional (default=None)
            Maximum scale value.
        permax : float, optional (default=None)
            Maximum period value.
        scut : float, optional (default=None)
            Scale cut-off value.
        pcut : float, optional (default=None)
            Period cut-off value.
        
        Notes
        -----
        This method defines the scales and converts them to periods or frequencies based on the wavelet type.
        It uses the wavelet's Fourier factor to perform the conversion.
        '''
        self.scales, self.params['scut'] = wt_utils.define_scales(
            self.wavelet['fourier_fac'],
            **{key: self.params[key] for key in ['dt', 'dj', 'n', 'scaletype', 'dj_lin']},
            s0=smin, permin=permin, permax=permax, smax=smax, pcut=pcut, scut=scut)
        
        # Convert to period/frequency
        self.convert_scales()

    def convert_scales(self):
        '''
        Convert scales to periods/frequencies
        
        Notes
        -----
        This method converts the scales to periods or frequencies based on the wavelet type.
        It uses the wavelet's Fourier factor to perform the conversion.
        '''
        # Convert scales to periods and frequencies
        self.periods, frequencies, periods_with_units = wt_utils.convert_scales(self.scales, self.wavelet['fourier_fac'], 
                                                                       self.params['dt_fct'])

        # Assign periods or frequencies
        if self.params['dt_fct'] is not None: 
            self.fper = periods_with_units
        elif self.params['fs'] is None:
            self.fper = self.periods
        else:
            self.fper = self.frequencies = frequencies


    def signal_and_padding(self, y: np.ndarray):
        '''
        Shaping and padding signal
        
        Parameters
        ----------
        y : numpy.ndarray
            Input signal to analyze.
            
        Notes
        -----
        This method shapes and pads the input signal for analysis.
        It removes the mean of the signal and applies padding to ensure
        the signal length is a multiple of 2.
        '''
        # Detrending
        y = y.copy() # Creating a copy not to modify the original signal
        y -= np.mean(y)

        # Define padding functions
        if not 'apply_padding' in self.params:
            self.params['n'] = len(y)
            self.set_padding()
        
        # Apply padding
        if self.params['apply_padding'] is not None:
            y = self.params['apply_padding'](y)

        # Reshaping
        y = np.reshape(y, (1, -1)) 

        return y

    def set_padding(self):
        '''
        Set padding functions (apply and remove)
            
        Notes
        -----
        This method defines padding functions to apply desired padding to 
        signals and remove padding from wavelet coefficients.
        '''
        # Padding
        self.params['apply_padding'], self.params['remove_padding'], self.params['n_ext'] = data_utils.sig_length_and_padding(self.params['n'], self.params['pad'], self.params['padmode'])

    def compute_WT(self, y_ext: np.ndarray):
        '''
        Define wavelet and compute wavelet coefficients using fft with parallelization and/or memory management
        
        Parameters
        ----------
        y_ext : numpy.ndarray
            Input (padded) signal to analyze.
        
        Notes
        -----
        This method is compatible with pyfftw and parallel processing.
        It uses pyfftw if installed and requested (`use_pyfftw` parameter) for faster FFT computation.
        The `parallel_proc` or `workers` parameter allow for parallel processing.
        If `chunk_size` is specified, it will process the signal in chunks.
        This is useful for processing large datasets that do not fit in memory.
        '''
        # Compute wavelet
        if not hasattr(self, 'psi'):
            self.psi = wt_utils.compute_psi(self.params['n_ext'], self.params['dt'], self.wavelet, self.scales)

        # Compute wavelet coefficients     
        self.wave = wt_utils.compute_wavelet_coef(y_ext, self.psi, self.processing['use_pyfftw'],
                                                  self.processing['parallel_proc'], self.processing['workers'],
                                                  self.processing['chunk_size'])   
    
    def compute_coi(self, 
                   coi_type: str = 'analytic', 
                   metric: str = 'int'):
        '''
        Compute the cone of influence (coi)
        
        Parameters
        ----------
        coi_type : str, optional (default='analytic')
            Type of cone of influence computation. 
            - 'analytic': based on area outside signal borders
            - 'numeric': based on dirac impulses on signal borders
        metric : str, optional (default='int')
            Metric for cone of influence computation.
        
        Notes
        -----
        This method computes the cone of influence based on the wavelet transform.
        It can be computed in two ways:
        1. Based on dirac impulses on signal borders (if required)
        2. Based on area outside signal borders
        '''
        percentout = self.params['percentout']
        n = self.params['n']
        # Computation type
        # Based on dirac impulses on signal borders (if required)
        if str(coi_type).startswith('num'): 
            self.coi = wt_utils.define_coi_dirac(self.wavelet, n, self.params['ts'], self.scales, percentout, metric)
        
        # Based on area outside signal borders
        dt = self.params['dt']
        fct = self.params['dt_fct']
        fper = self.fper
        if self.params['fs'] is None:
            unit = 'p'
        else:
            unit = 'f'
        self.coi = wt_utils.define_coi(self.wavelet, n, dt, percentout, fct, unit, fper, metric)

    def compute_signif(self, 
                       y: np.ndarray, 
                       lag1: Optional[float] = None):
        '''
        Compute the significance of the wavelet transform
        
        Parameters
        ----------
        y : array_like
            Signal to analyze.
        lag1 : float, optional (default=None)
            Lag parameter for significance computation.
        
        Notes
        -----
        This method computes the significance of the wavelet transform.
        It first computes the wavelet transform of the signal and then
        calculates the significance based on the power spectrum.
        '''
        self.signif = wt_utils.compute_significance(y, self.wavelet, self.params['dt'], 
                                            self.wave, self.periods, lag1)
    
    def xwt(self, 
            y2: Union["ArrayLike", "pd.Series", "pd.DataFrame"],
            y1: Optional[Union["ArrayLike", "pd.Series", "pd.DataFrame"]] = None):

        '''
        Compute the cross wavelet transform (xwt)
        
        Parameters
        ----------
        y2 : array_like
            Second signal to analyze.
        y1 : array_like, optional (default=None)
            First signal to analyze. If None, use the signal passed in the constructor.
        
        Notes
        -----
        This method computes the cross wavelet transform between two signals.
        It first computes the wavelet transform of each signal and then multiplies
        the complex conjugate of the first (y2) signal's wavelet coefficients with the second (y1) signal's wavelet coefficients.
        '''
        # Wavelet transform of y1
        if y1 is None:
            if not hasattr(self, 'wave'):
                raise AttributeError('Need to compute cwt for a signal before assessing cross wavelet transform')
        else:
            self.cwt(y1)

        # Wavelet transform of y2
        # Extract y2
        y2, *_ = data_utils.extract_signal_and_time(y2)   
         
        # Signal length and padding
        if len(y2) != self.params['n']:
            raise ValueError('Signal length (y2) must be equal to the length of the signal passed in the constructor (y1)')
        y2_ext = self.signal_and_padding(y2)

        # Compute wavelet coefficients     
        self.wave2 = wt_utils.compute_wavelet_coef(y2_ext, self.psi, self.processing['use_pyfftw'],
                                                  self.processing['parallel_proc'], self.processing['workers'],
                                                  self.processing['chunk_size'])   

        # Remove padding
        if self.params['remove_padding'] is not None:
            self.wave2 = self.params['remove_padding'](self.wave2)

        # # Significance
        # if self.flags['signif']: 
        #     self.signif2 = wt_utils.compute_significance(y2, self.wavelet, self.params['dt'], 
        #                                     self.wave2, self.periods, lag1)
        
        # Cross wavelet transform
        self.cross_wave = self.wave*np.conjugate(self.wave2)
        
    def scalogram(self, 
                  xdata: Optional[Union[np.ndarray, "pd.Series", "pd.DatetimeIndex", "DatetimeArray"]] = None, 
                  Units: Optional[str] = None):
        '''
        Compute the scalogram
        
        Parameters
        ----------
        xdata : array_like, pandas.Series, pandas.DatetimeIndex or pandas.arrays.DatetimeArray, optional (default=None)
            x-axis data for plotting. If None, x-axis data is np.arange(signal length).
        Units : str, optional (default=None)
            Units for x-axis. If None, x-axis is adimensional.
        '''
        
        
        coi = None if not hasattr(self, 'coi') else self.coi
        signif = None if not hasattr(self, 'signif') else self.signif
        
        if self.params['fs'] is None:
            wt_utils.plot_scalogram(self.wave, self.fper, self.params['ts'], xdata, coi, 'p', signif, Units)
        else:
            wt_utils.plot_scalogram(self.wave, self.fper, self.params['fs'], xdata, coi, 'f', signif, Units)

    def compute_mean_coef(self,
                          pstr: Optional[Union[str, List[str]]] = None,
                          pval: Optional[Union[int, float, "ArrayLike", "pd.Series", "pd.DataFrame"]] = None,
                          pspread: Optional[Union[int, float, "ArrayLike", "pd.Series", "pd.DataFrame"]] = None,
                          units: str = 'days'):
        '''
        Compute mean wavelet coefficients and power values for specified periods
        
        Parameters
        ----------
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
        '''
        
        if self.params['dt_fct'] is None:
            periods = self.periods
        else:
            periods = self.fper
            
        if hasattr(self, 'coi'):
            coi = self.coi
        else:
            coi = None
        
        if not hasattr(self, 'wave'):
            raise AttributeError('wave attribute not found. Please run cwt first.')

        str2val, coef, sgn, _, coefcoi, sgncoi = wai_utils.coefvalues(self.wave, periods, self.params['ts'] , coi, pstr, pval, pspread, units)
        self.mean_coef = {'power': coef, 'wave': sgn, 'powercoi': coefcoi, 'wavecoi': sgncoi}
        if hasattr(self, 'str2val'):
            self.str2val.update(str2val)
        else:
            self.str2val = str2val


    def compute_peaks(self,
                      x_val: Union["ArrayLike", "pd.Series", "pd.DatetimeIndex", "DatetimeArray"],
                      pstr: Union[str, List[str]],
                      filter_val: Optional[Union[
                          "pd.DataFrame",
                          Dict[str, Union[int, float, Sequence[Union[int, float]]]],
                          Union[int, float, Sequence[Union[int, float]]]
                      ]] = None,
                      filter_per: Optional[str] = None,
                      per: str = 'year',
                      thresh: Optional[Union[int, float]] = -np.inf,
                      coi: Optional[bool] = None):
        '''
        Compute peaks from wavelet coefficients
        
        Parameters
        ----------
        x_val : array_like, pandas.Series, pandas.DatetimeIndex, or pandas.arrays.DatetimeArray
            x-axis values for the signal
        pstr : str
            Period string identifier
        coi : bool, optional (default=None)
            If True, use cone of influence (COI) to remove edge effects for peak detection
        filter_val : dict, optional (default=None)
            Dictionary containing filter values for peak detection
        filter_per : str, optional (default=None)
            Filter period for peak detection
        per : str, optional (default='year')
            Period unit for peak detection
        thresh : int, float, optional (default=-inf)
            Threshold for peak detection
        '''
        # Check if COI flag is provided
        if coi is None:
            coi = self.flags['coi']

        # Check if coef attribute is available
        if not hasattr(self, 'mean_coef'):
            raise AttributeError('coef attribute not found. Please run compute_mean_coef first.')        

        # Extract coefficients and sign
        if coi:
            coef = self.mean_coef['powercoi']
            sgn = self.mean_coef['wavecoi']
        else:
            coef = self.mean_coef['power']
            sgn = self.mean_coef['wave']
        
        peak, *_ = wai_utils.extract_peaks(coef, x_val, pstr, filter_val, filter_per, per, sgn, thresh)
        
        if not hasattr(self, 'peak'):
            self.peak = {}

        self.peak[self.wavelet['name']] = peak

    def WAI(self,
            idx: Union[int, float, "timedelta", "ArrayLike", "pd.Series", "pd.DatetimeIndex", "DatetimeArray"],
            per: Union[int, float, "timedelta", str],
            data: Union["ArrayLike", "pd.Series", "pd.DataFrame"],
            xdata: Optional[Union["ArrayLike", "pd.Series", "pd.DatetimeIndex", "DatetimeArray"]] = None,
            wname: Optional[str] = None, 
            perpeak: Optional[str] = None,
            filter_per: Optional[str] = None,
            filter_dates: Optional[Union[int, float, "ArrayLike", "pd.Series"]] = None,
            plotting_mode: str = 'together',
            title: Optional[str] = 'default',
            xlims: Optional[Union["ArrayLike", "pd.Series", "pd.DatetimeIndex", "DatetimeArray"]] = None):
        '''
        Visualize wavelet coefficient construction and Wavelet Area Interpretation (WAI)
        
        Parameters
        ----------
        idx : int, float, timedelta, array_like, pandas.Series, pandas.DatetimeIndex, or pandas.arrays.DatetimeArray
            Indices or positions to analyze, either as a value, timedelta, or string identifier.
            If (1) per is not a string, or (2) perpeak is not specified, or (3) no peak dataframe is provided, 
            idx indicates the indices (x-values) of the signal to be analyzed.
            Otherwise, idx must indicate idx values to be used from the chosen peak dataframe:
            For all/sup peaks, it must correpond to values in column 'idx'
            For repeated peaks, it must correspond to index values of the peak dataframe
        per : int, float, timedelta, str
            Period to analyze, either as a value, timedelta, or string identifier.
            If str, str2val must indicate the value of the period, either by providing
            the value directly or by providing a dictionary with the mapping (output of coefvalues)
        data : array_like, pandas.Series, or pandas.DataFrame
            The series data. If a DataFrame/Series is given, xdata (if missing) values are taken from the index
            or from columns 'xdata' or 'time'/'datetime' (if datetime)
        xdata : array_like, pandas.Series, pandas.DatetimeIndex, or pandas.arrays.DatetimeArray, optional (default=None)
            The x-axis values corresponding to the data. If None and no xdata extracted from data, 
            xdata is set to np.arange(len(data)).
        perpeak : float or str, optional
            Period to use for peak visualization, among field names in peaks ('all', 'sup' 
            and identifiers for repeated peaks, e.g. 'apr', 'jun' or 'nov').
        filter_per : str, optional
            Period used for filtering.
            Ignored if xdata is not a datetime.
        filter_dates : list of int or float, optional
            Value used for filtering peaks.
            Ignored if xdata is not a datetime.
        plotting_mode : str, default='together'
            Plotting mode: 'together' or 'separate'
        title : str, default='default'
            Title for the plot
        '''

        # Check if peak attribute is available
        if not hasattr(self, 'peak'):
            raise AttributeError('peak attribute not found. Please run compute_peaks first.')
        
        # Check if wavelet name is provided and convert to lowercase
        if wname is None:
            wname = self.wavelet['name']
            wavelet = self.wavelet
        else:
            wname = str(wname).lower()
            wavelet = wname
        
        # Check if wavelet name is in peak attribute
        if perpeak is not None:
            if not wname in self.peak:
                raise ValueError(f'Wavelet {wname} not found in peak attribute. '
                                'Please compute wavelet coefficients first for this wavelet.')

            # Extract peak dataframe
            peak = self.peak[wname]
        else:
            peak = None
        
        # Launch WAI
        wai_utils.launch_wai(idx, wavelet, per, self.params['ts'], data, xdata, self.str2val, self.str2val['units'],
                            peak, perpeak, filter_dates, filter_per, plotting_mode, title, xlims)
        
    def translate_peak(self,
            peak_in: Union[int, float, "ArrayLike", "pd.Series", "pd.DataFrame"],
            per: Union[str, int, float, "timedelta"],
            wname: Optional[str] = None
        ) -> Union[int, float, np.ndarray, "pd.Series", "pd.DataFrame"]:
        '''
        Find indices of extrema or zero crossings from wavelet center indices.
        
        Parameters
        ----------
        peak_in : int, float, array_like, pandas.Series, pandas.DataFrame
            Position(s) of the wavelet coefficient(s) to be translated
            If dataframe with more than 1 column, f"{per}loc" must indicate the column to consider.
        per : str, int, float, timedelta
            Period to analyze, either as a value, timedelta, or string identifier.
            If str, attribute str2val must indicate the value of the period (output of coefvalues)

        Returns
        -------
        idxs : int, float, np.ndarray, pandas.Series, pandas.DataFrame
            Indices of extrema or zero crossings
        '''

        # Check if wavelet name is provided and convert to lowercase
        if wname is None:
            wavelet = self.wavelet
        else:
            wavelet = str(wname).lower()

        # Check if str2val attribute is available
        if hasattr(self, 'str2val'):
            return wai_utils.idxs_from_center_to_extremum_or_zero_dog(peak_in, per, self.params['ts'], wavelet, self.str2val)
        else:
            return wai_utils.idxs_from_center_to_extremum_or_zero_dog(peak_in, per, self.params['ts'], wavelet)
 
    def support_period(self,
            per: Union[int, float, "timedelta"],
            percentout: float = 0.01,
            wname: Optional[str] = None,
            metric: str = 'int'
        ) -> Union[int, float, "timedelta"]:
        '''
        Compute the support for a given wavelet and period.
        
        Parameters
        ----------
        per : int, float, timedelta
            Period of the wavelet
        percentout : float, optional (default=0.1)
            Percentage of metric allowed outside the support
        metric : str, optional (default='int')
            Metric to compute percentout, based on absolute wavelet coefficients ('int' or 'integral') 
            or squared wavelet coefficients ('ene' or 'energy').
        
        Returns
        -------
        support : int, float or timedelta
            The computed support in the same units as the input period
        '''

        # Check if wavelet name is provided and convert to lowercase
        if wname is None:
            wavelet = self.wavelet
        else:
            wavelet = str(wname).lower()

        return wai_utils.compute_support_period(wavelet, per, self.params['ts'], percentout, metric)

