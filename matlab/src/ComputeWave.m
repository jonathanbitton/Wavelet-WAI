function Tb = ComputeWave(var,time,periods,scales,dt,ai,bi,ftype,varargin)
% Computing wavelets & coefficients at defined period and date

%%% Inputs
% - var (double): signal
% - time (datetime or double): time (or other, e.g. space) data corresponding to each signal data point
% - periods (duration): periods ('Ts' specified) OR frequencies ('Fs' specified)
% - scales (double): scales used (before conversion to periods or frequencies)
% - dt (double): Adimensionnal sampling period
% - ai (duration or double): period to compute wavelet. Same units as 'periods'
% - bi (datetime or double): time to compute wavelet. Same units as 'time'
% - ftype (char): Analyzing function used among 'Morlet', 'Mexhat', 'DOGx', 'Gauss', 'Haar'
% Optional arguments using the format (...,'Argument',Value,'Argument',Value):
% 1) 's2' (double): Variance of the Gaussian function used to define wavelets
% 2) 'w0' (double): Central angular frequency of the Morlet wavelet (ignored if ftype not set to Morlet)
% Default settings are s2=1 and w0=6

%%% Outputs
% Tb: table containing signal, wavelet and positive/negative areas

% Importing data
% Extract method
s2 = 1;
if any(strcmpi(varargin,'s2'))
    ind = find(strcmpi(varargin,'s2'));
    s2 = (varargin{ind+1});
    if ~isnumeric(s2)
        error("s2 must be a double, not a %s",class(s2))
    end
end

% Extract percentout
w0 = 6;
if any(strcmpi(varargin,'w0'))
    ind = find(strcmpi(varargin,'w0'));
    w0 = varargin{ind+1};
    if ~isnumeric(w0)
        error("w0 must be a double, not a %s",class(w0))
    end
end


% Initilaizing
Tb = timetable(time,var); % Creating output table
L = length(var); % Number of data points
t = (1:L)'; % Time vector (adimensional)
isComplex = false; % Complex wavelet flag
% Convert ftype and verify formats
[ftype,dt] = GetArgs({'ftype','dt'},'ftype',ftype,'ts',dt);

% Period
[~,a] = min(abs(periods-ai)); % Location of period studied
a = scales(a); % Conversion from period to scale

% Time
b = find(time==bi); % Time position


% Defining functions/wavelets for specified time and scale
switch ftype(1:3)
    case 'MOR'
        isComplex = true;
        Tb.(strcat(ftype,"_re"))=pi^(-0.25)*cos(w0*dt*(t-b)/a).*exp(-(dt*(t-b)/a).^2*0.5/s2); % pi^(-0.25) for Norm L2 (arbitrary)
        Tb.(strcat(ftype,"_im")) = pi^(-0.25)*sin(w0*dt*(t-b)/a).*exp(-(dt*(t-b)/a).^2*0.5/s2);
        Tb.(ftype) = pi^(-0.25)*exp(w0*1i*dt*(t-b)/a).*exp(-(dt*(t-b)/a).^2*0.5/s2);
    case 'DOG'
        m = sscanf(ftype,'DOG%d'); 
        syms xvar
        Expr=matlabFunction((-1)^(m+1)*diff(exp(-xvar^2/2/s2),m));
        Tb.(ftype) = Expr((t-b)*dt/a)/sqrt(gamma(m+0.5));
    case 'HAA'
        psi = zeros(length(t),1);
        psi(t>=-0.5*a/dt+b & t<b) = 1;
        psi(t>=b & t<0.5*a/dt+b) = -1;
        Tb.(ftype) = psi;
    case 'GAU'
        Tb.(ftype) = exp(-(dt*(t-b)/a).^2*0.5/s2)/sqrt(s2*gamma(0.5));
end
% Product curves and areas
Tb.(strcat("Res",ftype))=Tb.(ftype).*Tb.var; % Product curve
Tb.(strcat("PosRes",ftype,"_re")) = max(real(Tb.(strcat("Res",ftype))),0); % Positive Area (real)
Tb.(strcat("NegRes",ftype,"_re")) = min(real(Tb.(strcat("Res",ftype))),0); % Negative Area (real)
if isComplex
    Tb.(strcat("PosRes",ftype,"_im")) = max(imag(Tb.(strcat("Res",ftype))),0); % Positive Area (imag)
    Tb.(strcat("NegRes",ftype,"_im")) = min(imag(Tb.(strcat("Res",ftype))),0); % Negative Area (imag)
end
