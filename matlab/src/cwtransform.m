function varargout = cwtransform(y,varargin)
% Continuous Wavelet Transform of signal y

%%% Inputs:
% - y (double): Signal to analyze using the CWT
% Optional arguments of the transform, using the format 
% cwtransform(y,'Argument',Value,'Argument',Value,...). Options available:
% 1) 'ftype' (char): Analyzing function used among 'Morlet', 'Mexhat', 'DOGx', 'Gauss', 'Haar'
% 2) 'Ts' (duration or double): Sampling period (temporal resolution)
% 2b) 'Fs' (double): Sampling frequency (temporal resolution)
% 3) 'dj' (double): Frequency resolution
% 4) 'permin' (double): Lowest period studied
% 4b) 'fmax' (double): Highest frequency studied
% 4c) 'smin' (double): Lowest scale studied
% 5) 'permax' (double): Highest period studied
% 5b) 'fmin' (double): Lowest frequency studied
% 5c) 'smax' (double): highest scale studied
% 6) 'norm' (char): Normalization convention, among 'L1' and 'L2'
% 7) 'pad' (char): Padding method: 'zpd (zeros), 'rfl' (reflection at
% signal edges), 'rfle' (reflection without boundary duplicates), 
% 'per' (periodic) or 'none'
% 8) 'padmode' (char): Edge of the signal to pad: 'left', 'right' or 'both'
% 9) 'nosig': no significance computation, no value needed for this argument
% 10) 's2' (double): Variance of the Haar or Gaussian function used to define wavelets
% 11) 'w0' (double): Central angular frequency of the Morlet wavelet (ignored if ftype not set to Morlet)
% 12) 'percentout' (double): Threshold (<1) to define coefficients affected by edges
% 13) 'coimethod' (char): Method of COI computation: analytic ('anal') or numerical ('num') resolution (ignored if ftype not set to DOGx)
% 14) 'plot': Plot the scalogram, no value needed for this argument
% 15) 'time' (double or datetime): xaxis for the plot
% 16) 'plotunits' (char): Units for the plot

% To preserve units, 'Ts' must be a duration (only versions>=R2014b)
% By specifying Ts(Fs), outputs will be converted to periods(frequencies)

% Arguments mutually exculsive (cannot be entered as options together) :
% 'Ts' & 'Fs'
% 'permin' & 'fmax' & 'smin'
% 'permax' & 'fmin' & 'smax'

% Default argument values:
% 1) 'ftype': Morlet wavelet
% 2) 'Ts': 1 (unitless)
% 2b) 'Fs': not used (Ts specified)
% 3) 'dj': 1/20
% 4) 'permin': 2 * time resolution
% 4b) 'fmax': not used ('permin' specified)
% 4c) 's0': not used ('permin' specified)
% 5) 'permax': length of singal * time resolution
% 5b) 'fmin': not used ('permax' specified)
% 5c) 'smax': not used ('permax' sepcified)
% 6) 'norm': 'L1'
% 7) 'pad': Symmetric padding (half point)
% 8) 'padmode': 'both'
% 9) 'nosig': active if plot is not required & output 'signif' is not desired 
% 10) 's2': 1
% 11) 'w0': 6 (=> central frequency Fc set to 6/2/pi~=1)
% 12) 'percentout': 0.02 (e^(-2) also commonly used)
% 13) 'coimethod': 'anal'
% 14) 'plot': active if the function is called without specifying outputs
% 15) 'time': (1:numel(y)) * Ts
% 16) 'plotunits': same as Ts

%%% Outputs :
% wave: array of wavelet coefficients for each couple (time,period)
% power: power of wavelet coefficients, computed as (abs(wave))^2
% periods: periods ('Ts' specified) OR frequencies ('Fs' specified)
% scales: scales used (before conversion to periods or frequencies)
% COI: cone of influence (1 indicates coefficients affected by edges)
% signif: significant wavelet coefficients (>1)

% References: 
% Torrence, C. and G. P. Compo, 1998: A Practical Guide to Wavelet Analysis. 
% Bull. Amer. Meteor. Soc.</I>, 79, 61-78.

if nargin>31
    error('Too many input arguments')
end

% Signal conversion (values in lines)
y = y(:)';

% Parameters of the transform
DataOut = {'ftype','Ts','dt','dtf','Fs','dj','Period_Inf','s0','Period_Sup','s_max','norm','pad',...
    'padmode','s2','w0','percentout','coimethod','Figure','signif','time','Units'};
[ftype,Ts,dt,fct,Fs,dj,PerInf,s0,PerSup,s_max,norm,pad,padmode,s2,w0,...
    qcoi,COItype,Figure,signif,time,Units] = GetArgs(DataOut,varargin{:});

%% Parameters for each wavelet and normalization
switch ftype(1:3)
    case 'MOR' % Morlet wavelet
        if strcmp(norm,'L1') % L1 norm
            Fact_Fourier = 2*pi/w0; % Fourier Factor, defined as the ratio of period to scale
            cst = [2, s2/2, w0]; % Parameters for Morlet, used for future computations
        else % L2 norm
            Fact_Fourier = 4*pi*sqrt(s2)/(sqrt(s2)*w0+sqrt(2+s2*w0^2)); 
            cst = [(4*pi*s2)^(1/4), s2/2, w0];
        end
        cst2 = (4*pi*s2)^(1/4)/cst(1); % Norm conversion L1 to L2
        s_wav = 2*s2; % 2*Variance (used for COI)
    case 'GAU' % Gauss function
        NormL1 = 2; % Value of the L1 Norm
        Fact_Fourier = 2*pi*sqrt(s2*2); % Only defined for energy
        if strcmp(norm,'L1')
            cst = [NormL1, s2/2]; % NormL1/sqrt(exp(1)*s2/4)*sqrt(2pi*s2)
        else
            cst = [sqrt(2*sqrt(pi*s2)), s2/2];
        end
        cst2 = sqrt(2*sqrt(pi*s2))/cst(1);
        s_wav = 2*s2;
    case 'DOG' % Gaussian derivatives 
        m = sscanf(ftype,'DOG%d'); % Order of the derivative
        NormL1 = 2; 
        if strcmp(norm,'L1')
            Fact_Fourier = 2*pi*sqrt(s2)/sqrt(m);
            cst = [nan, s2/2, m];
            if m<6
                vals = [2 4/sqrt(exp(1)) 8/sqrt(exp(3))+2 4*sqrt(6)/exp(1.5+sqrt(1.5))*(sqrt(3+sqrt(6))+sqrt(3-sqrt(6))*exp(sqrt(6))) 2/exp(2.5+sqrt(2.5))*(16+8*sqrt(10)+(8*sqrt(10)-16)*exp(sqrt(10)))+6]; 
                cst(1) = vals(m)/s2^((m-1)/2);
            else
                syms xvar
                vals2 = matlabFunction(abs(diff(exp(-xvar^2/2/s2),m)));
                cst(1) = integral(vals2,-Inf,Inf,'RelTol',1e-15,'AbsTol',1e-15);
            end
            cst(1) = -(1i)^m*NormL1/cst(1)*sqrt(2*pi*s2);
        else 
            Fact_Fourier = 2*pi*sqrt(s2)/sqrt(m+1/2);
            cst = [-(1i)^m*sqrt(s2^(m-0.5)/gamma(m+0.5))*sqrt(2*pi*s2), s2/2, m];
        end
        cst2 = -(1i)^m*sqrt(s2^(m-0.5)/gamma(m+0.5))*sqrt(2*pi*s2)/cst(1);
        meth = 'integral';
        s_wav = (2*s2)*(1+log(2)/log(qcoi)); % only used if m=1 (DOG1)
    case 'HAA' % Haar wavelet
        NormL1 = 2;
        if strcmp(norm,'L1')
            Fact_Fourier = 2*pi/4.6622447;
            cst = NormL1;
        else
            Fact_Fourier = 2*pi/5.5729963;
            cst = 1;
        end
        cst2 = 1/cst;
end

%% Padding
x = y - mean(y); % Detrending (optional)
n1 = length(y); % Length before padding

% Padding to the nearest higher power of 2 (if needed)
if ~isequal(pad,'none')
    base = fix(log2(n1) + 0.4999); % .5 and below => round to lower exponent
    % Right padding
    if strcmp(padmode,'r')
        ext = fix(2^(base+1)-n1); % Padding length
        if strcmp(pad,'zpd') % zero padding
            x = [x, zeros(1,ext)];
        elseif strcmp(pad,'rfl') % symetric extension at boundaries
            x = [x, x(end:-1:end-ext+1)];
        elseif strcmp(pad,'rfle') % symetric extension (no boundary duplicates)
            x = [x, x(end-1:-1:end-ext)];
        elseif strcmp(pad,'per') % periodic extension
            x = [x, x(1:1:ext)];
        end
    % Left padding
    elseif strcmp(padmode,'l')
        ext = fix(2^(base+1)-n1);
        if strcmp(pad,'zpd')
            x = [zeros(1,ext), x];
        elseif strcmp(pad,'rfl')
            x = [fliplr(x(1:ext)), x];
        elseif strcmp(pad,'rfle')
            x = [fliplr(x(2:ext)), x];
        elseif strcmp(pad,'per')
            x = [x(end-ext+1:1:end), x];
        end
    % Left and right padding
    elseif strcmp(padmode,'b') 
        ext = (2^(base)-n1/2);
        if strcmp(pad,'zpd')
            x = [zeros(1,ceil(ext)), x, zeros(1,fix(ext))];
        elseif strcmp(pad,'rfl')
            x = [fliplr(x(1:ceil(ext))), x, x(end:-1:end-fix(ext)+1)];
        elseif strcmp(pad,'rfle')
            x = [fliplr(x(2:ceil(ext)+1)), x, x(end-1:-1:end-fix(ext))];
        elseif strcmp(pad,'per')
            x = [x(end-ceil(ext)+1:1:end), x, x(1:1:fix(ext))];
        end
    end
end

n = length(x); % Signal length after padding (if required)

%% Scales
% Lowest and highest scale
[s0,s_max]=DefineScales(dt,n1,Fact_Fourier,PerInf,PerSup,s0,s_max);

% Maximal Index for scales
J = fix(log2(s_max/s0)/dj);

% Scales for which wavelets are computed
scales = s0*2.^((0:J)*dj);

% Converting scales to periods/frequencies using the Fourier factor
fper = Fact_Fourier*scales'; % periods analyzed
periods = fper; % unitless copy of the value
if ~isempty(fct) % add units if 'Ts' is a duration 
    fper = fct(periods); 
    fper.Format = Ts.Format; % matching formats 
elseif ~isempty(Fs) % convert to frequencies if Fs specified
    fper = 1./fper;
end

%% Continuous Wavelet Transform computation
% Fourier Transform of signal
y_ft = fft(x);

% Angular frequencies
if rem(n,2)  % odd sample size 
    w = 1:fix(n/2); % exploiting symetrical property of FT
    w = [0, w, -fliplr(w)]*(2*pi/(n*dt));
elseif ~rem(n,2)  % even sample size
    w = 1:n/2;
    w = [0, w, -fliplr(w(1:end-1))]*(2*pi/(n*dt));
end

% Computation of wavelet coefficients
wave = Computewt(ftype,cst,scales,w,dt,y_ft,norm);

% Remove padding if applicable
if ~isequal(pad,'none')
    if strcmp(padmode,'r')
        wave = wave(:,1:n1);
    elseif strcmp(padmode,'l')
        wave = wave(:,n1+1:end);
    elseif strcmp(padmode,'b')
        wave = wave(:,ceil(ext)+1:ceil(ext)+n1);
    end
    n = length(y);
end 

% Power computation
power = (abs(wave)).^2 ;

%% Cone of Influence
if strcmp(ftype(1:3),'DOG') && m~=1 && (nargout > 4 || nargout ==0)
    if strcmp(COItype,'num') 
        COI = DiracCOIComputation(ftype,n,Ts,scales,qcoi);
    else
        coival = DOGCOI(meth,m,s2,Fact_Fourier,qcoi);
    end
elseif strcmp(ftype,'HAA')
    coival = Fact_Fourier/(0.5-qcoi);
else
    coival = Fact_Fourier/sqrt(s_wav*(-log(qcoi)));
end

if exist('coival','var')
    if rem(n,2)  % odd sample size 
       indices = 1:ceil(n/2); % indices for signal values
       indices = [indices, fliplr(indices(1:end-1))]; % COI symetric w.r.t center
    elseif ~rem(n,2)  % even sample size 
       indices = 1:n/2;
       indices = [indices, fliplr(indices)];
    end
    % Computing the period for which coefficients are affected at each time
    COI = coival*dt*indices; 
    if ~isempty(fct) % if Ts is a duration
        COI = fct(COI); % adding units
        COI.Format = Ts.Format; % matching formats
    end
    % Adjusting the COI to borders for frequencies above the Nyquist
    if isempty(Fs)
        COI(COI<2*Ts) = min(fper);
    else
        COI(COI<2/Fs) = min(periods);
        COI = 1./COI;
    end
end

%% Significance
% Plot scalogram if no output required
if nargout==0, Figure = true; end

% Significance
if signif && (nargout > 5 || Figure)
    signif = significance(periods,dt,ftype,y,'prob',0.95); % threhold for significance for each period
    signif = repmat(signif(:),1,n); % repeat for eah line
    if strcmp(norm,'L1') % Conversion of power to match theoretical spectrum
        powersig = wave*cst2; % Norm L2 (energy of mother wavelet) = 1
        powersig = ((scales'/dt).^0.5).*powersig; % Conservation of norm L2 for daughter wavelets
        powersig = (abs(powersig)).^2 ; % Power
        signif = powersig./signif; % >1 = significant coefficient
    else
        signif = power./signif;
    end
else
    signif = [];
end

%% Scalogram
if Figure
    if isempty(time)
        time = 0:dt:numel(y)*dt-dt; % Creation of time vector
    else
        if ~isequal(numel(time),numel(y))
            error('time and signal dimensions do not match')
        end
    end
    if isempty(Fs)
        plotscalogram(wave,fper,time,COI,Ts,'signif',signif,'plotunits',Units)
    else
        plotscalogram(wave,fper,time,COI,Ts,'signif',signif,'plotunits',Units,'FP','f')
    end
end

%% Outputs
varargout = {wave,power,fper,scales,COI,signif};
varargout = varargout(1:nargout);



function [s0,s_max] = DefineScales(dt,n1,Fact_Fourier,PerInf,PerSup,s0,s_max)
% Lowest scale
if isempty(s0)
    if isempty(PerInf)
        PerInf = 2*dt; % Default
    else % fmax or permin specified
        if PerInf < 2*dt % Lower bound (for interpretation)
            warning('permin(fmax) does not respect the %s frequency. \n A value of 2*Ts(Fs/2) is recommended','Nyquist')
        end
    end
    s0 = PerInf/Fact_Fourier; % Converting periods to scales using the Fourier factor
elseif s0*Fact_Fourier < 2*dt
    warning('smin does not respect the %s frequency. \n A value of 2*Ts(Fs/2) for permin(fmax) is recommended','Nyquist')
end

% Highest scale
if isempty(s_max)
    if isempty(PerSup) 
        PerSup = n1*dt; % Default
    else
        if PerSup > n1*dt % Upper bound (signal length)
            warning('permax(fmin) exceeds the signal length')
        end
    end
    s_max = PerSup/Fact_Fourier; % Converting periods to scales using the Fourier factor
elseif s_max*Fact_Fourier > n1*dt
    warning('smax exceeds the signal length')
end
% Lowest scale analyzed must be inferior to the highest
if s_max < s0 
    error('permin>permax, smin>smax or fmax>fmin are inadmissible')
end


function wave = Computewt(ftype,cst,scales,w,dt,y_ft,norm)
switch ftype(1:3)
    case 'MOR' % Morlet
        Psi = cst(1)*exp(-cst(2)*(scales'*w-cst(3)).^2).*(w>0);
    case 'DOG' % DOG functions
        Psi = cst(1)*exp(-cst(2)*(scales'*w).^2).*((scales'*w).^cst(3));
    case 'GAU' % Gauss function (NOT a wavelet transform but a filtering)
        Psi = cst(1)*exp(-cst(2)*(scales'*w).^2);
    case 'HAA' % Haar
        Psi = -4*cst(1)*1i*(sin((scales'*w)/4)).^2./(scales'*w);
        Psi(:,w==0)=0;
end
if strcmp(norm,'L2') % If L2 Normalization applied
    Psi = ((scales'/dt).^0.5).*Psi;
end
wave = ifft(y_ft.*Psi,[],2); % Wavelet Coefficients

