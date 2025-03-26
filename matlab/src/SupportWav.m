function Support = SupportWav(ftype,ai,Ts,varargin)
% Yields Support of DOG and Gauss fonctions (in units of ai)

%%% Inpupts
% - 'ftype' (char): Analyzing function (wavelet)
% - 'ai' (duration): Period of analyzing function
% - 'Ts' (duration): Sampling period
% Optional arguments using the format (...,'Argument',Value,'Argument',Value,...):
% 1) 'method' (char): Metric to define support, 'integral' (area) or 'energy' (squared area)
% 2) 'norm' (char): 'L1' or 'L2'
% 3) 'percentout' (double): Percentage of (squared) area to omit for support
% 4) 's2' (double): Variance of the Gaussian function used to define wavelets


if nargin>11
    error('Too many input arguments')
end

% Extract method
method = 'int';
if any(strcmpi(varargin,'method'))
    ind = find(strcmpi(varargin,'method'));
    method = char(string(lower(varargin{ind+1})));
    if method<3
        error("Metric '%s' inadmissible for support computation. \nPlease select an option among 'intergal' or 'energy'",method)
    end
    method = method(1:3);
    if ~ismember(method,["int","ene"])
        error("Metric '%s' inadmissible for support computation. \nPlease select an option among 'intergal' or 'energy'",method)
    end
    varargin(ind+1)=[];
    varargin(ind)=[];
end

% Extract percentout
percentout = 0.1;
if any(strcmpi(varargin,'percentout'))
    ind = find(strcmpi(varargin,'percentout'));
    percentout = varargin{ind+1};
    if ~isnumeric(percentout)
        error("percentout must be a double, not a %s",class(percentout))
    elseif percentout>=1
        error('percentout value %s inadmissible. \nPlease specify a value inferior to 1',string(percentout))
    end
    varargin(ind+1)=[];
    varargin(ind)=[];
end

% Extract Fourier Factor and verify inputs
[Fact_Fourier,ftype,s2] = DefFourierF(ftype,varargin{:});%'norm',norm,'s2',s2);

% ai & Ts formats
if ~isduration(Ts)
    error("Ts must be a duration, not a %s",class(Ts))
else
    [dt,fct] = UnitsAndFctHandles(Ts); % Unitless sampling interval
end
if ~isduration(ai)
    error("ai must be a duration, not a %s",class(ai))
end

% Initialize
s = fct(ai)/Fact_Fourier;

if strcmp(ftype,'MOR'),ftype='GAU';end % Same value
switch(ftype(1:3))
    case 'GAU'
        I = erfinv(2*(0.5-percentout/2)); % Analytical solution of bound for 1-percentout of area of function e^(-x²)
        if strcmp(method,'int')
            I = I*sqrt(2*s2); % Conversion for e^(-x²/2/s2)
        else
            I = I*sqrt(s2); % Conversion for e^(-x^2/s2)
        end
    case 'DOG' % DOG
        m = sscanf(ftype,'DOG%d');
        coival = DOGCOI(method,m,s2,Fact_Fourier,percentout/2); % Exploiting COI function
        I = Fact_Fourier/coival; % Conversion
	case 'HAA' % Haar
		I = 1/2
end
I = I*s/dt; % Multiply by scale/dt for support limits of x-axis (edges of support)
% Support
Support = (fct(2*I*dt)); 


function [ftype,method,percentout,s2,m,Fact_Fourier,dt,fct] = Getargs(ftype,ai,Ts,varargin)

% Initial values
norm = [];
percentout = 0.1;
method = 'integral';
s2 = 1;

% Modifications according to inputs
if nargin>3
    nb = length(varargin);
    i = 1;
    while i<=nb
        name = lower(varargin{i});
        if i<nb
            value = varargin{i+1}; 
        end
        i = i+2;
        switch name
            case 'norm', norm = value;
            case 'percentout', percentout = value;
            case 'method', method = char(lower(value));
            case 's2', s2 = value;
            otherwise
                error('Error: Input Argument "%s" inadmissible. \nPlease refer to the top commented section of the function',name);
        end
    end
end

[ftype,Fact_Fourier] = FormatWavAndDefFourierF(ftype,norm,s2,w0);
% ftype = char(upper(ftype));
% 
% % Wavelet (ftype) format
% if ftype<3
%     error('Fonction "%s" inadmissible. \nPlease select an option among DOGx (x=order of derivation) or Gauss',ftype)
% end
% if strcmp(ftype(1:3),'DOG') % For DOG wavelets
%     if isempty(regexp(ftype, '^DOG\d+$', 'once'))
%         error('Derivation order of "%s" inadmissible. \nPlease sepcify a numeric derivation order',ftype)
%     end
%     % Conversion of DOG0 to Gauss
%     if sscanf(ftype,'DOG%d')==0, ftype = 'GAU';end
% end
% % Conversion of Mexican Hat to DOG2
% if strcmp(ftype(1:3),'MEX'), ftype = 'DOG2';end

% norm
if isempty(norm)
    norm='L1';
elseif isnumeric(norm) % Changing to char if numeric input
    if norm==2
        norm = 'L2';
    else
        norm = 'L1';
    end
end
% norm format
if ~ischar(norm) && ~isstring(norm)
    error("norm must be a char or string, not a%s",class(norm))
end
if ~ismember(norm,['L1','L2',"L1","L2"])
    error('Normalization method "%s" inadmissible. \nPlease select an option among L1 or L2',norm)
end

% method format
if method<3
    error("Metric '%s' inadmissible for support computation. \nPlease select an option among 'intergal' or 'Energy'",method)
end
method = method(1:3);
if ~ismember(method,['int','ene'])
    error('Normalization method "%s" inadmissible. \nPlease select an option among L1 or L2',norm)
end

% percentout format
if ~isnumeric(percentout)
    error("percentout must be a double, not a %s",class(percentout))
elseif percentout>=1
    error('percentout value %s inadmissible. \nPlease specify a value inferior to 1',string(percentout))
end

% s2 format
if ~isnumeric(s2)
    error("dj must be a double, not a %s",class(s2))
end

% ai & Ts formats
if ~isduration(Ts)
    error("Ts must be a duration, not a %s",class(Ts))
elseif ~isduration(ai)
    error("ai must be a duration, not a %s",class(ai))
end
% Adimensional sampling interval & funciton handle
[dt,fct] = UnitsAndFctHandles(Ts);

% Fourier Factor and derivation order
switch ftype(1:3)
    case 'DOG'
        m = sscanf(ftype,'DOG%d'); % order of derivation
        if strcmpi(norm,'L1')
            Fact_Fourier = 2*pi*sqrt(s2)/sqrt(m);
        else 
            Fact_Fourier = 2*pi*sqrt(s2)/sqrt(m+1/2);
        end
    case 'GAU'
        m = 0;
        Fact_Fourier = 2*pi*sqrt(s2*2);
    otherwise
            error('Fonction "%s" inadmissible. \nPlease select an option among DOGx (x=order of derivation) or Gauss',ftype)
end