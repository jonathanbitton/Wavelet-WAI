function varargout = GetArgs(DataOut,varargin)
% Formatting and storing inputs of functions

%%% Inupts
% Optional arguments using the format (...,'Argument',Value,'Argument',Value,...):
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
% 12) 'percentout' (double): Threshold (<1) to define coefficients affected by edge effect (COI or wavelet suport)
% 13) 'coimethod' (char): Method of COI computation: analytic ('anal') or numerical ('num') resolution (ignored if ftype not set to DOGx)
% 14) 'plot': Plot the scalogram, no value needed for this argument
% 15) 'time' (double or datetime): xaxis for the plot
% 16) 'plotunits' (char): Units for the plot

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

% Setting default and user-defined arguments

% Matlab version (for datetimes)
v = version('-release');
year = str2double(v(1:end-1));

% Initial values
ftype = 'MORLET';
Ts = [];
Fs = [];
dt = [];
dtf = [];
dj = 1/20;
Period_Inf = [];
Freq_Inf = [];
s0 = [];
Period_Sup = [];
Freq_Sup = [];
s_max = [];
norm = [];
pad  = 'rfl';
padmode = 'b';
s2 = 1;
w0 = 6;
pctout = 0.02;
coimethod = 'anal';
Figure = false;
signif = true;
time = [];
Units = [];

% Modifications according to inputs, format verification & conversions
if nargin>0
    nb = length(varargin);
    i = 1;
    while i<=nb
        name = lower(varargin{i});
        if i<nb
            value = varargin{i+1}; 
        end
        i = i+2;
        switch name
            case 'ftype', ftype = char(string(upper(value)));
                if ftype<3
                    error('Wavelet "%s" inadmissible. \nPlease select an option among Morlet, Mexhat, DOGx (x=order of derivation), Gauss or Haar',ftype)
                end
                if ~ismember(ftype(1:3),{'MOR','MEX','HAA','DOG','GAU'}) % Haar
                    error('Wavelet "%s" inadmissible. \nPlease select an option among Morlet, Mexhat, DOGx (x=order of derivation), Gauss or Haar',ftype)
                end
                if strcmp(ftype(1:3),'DOG') % Format of wavelets
                    if isempty(regexp(ftype, '^DOG\d+$', 'once')) % DOG
                        error('Derivation order of "%s" inadmissible. \nPlease sepcify a numeric derivation order',ftype)
                    end
                    ftype = strcat('DOG',num2str(sscanf(ftype,'DOG%d'))); % Convert for unnecessary digits
                else
                    ftype = ftype(1:3); % Others
                end
                % Conversion of Mexican Hat to DOG2
                if strcmp(ftype(1:3),'MEX'), ftype = 'DOG2';end
                % Conversion of DOG0 to Gauss
                if sscanf(ftype,'DOG%d')==0, ftype = 'GAU';end
            case 'ts', Ts = value;
                dt = Ts; % Adminesionnal sampling period
                if year>2014 || (year==2014 && v(end)=='b')
                    if ~isduration(Ts) && ~isnumeric(Ts) % Format of Ts
                        error("Ts must be a duration or double, not a %s",class(Ts))
                    end
                    if isduration(Ts) % Convert Ts
                        [dt,dtf] = UnitsAndFctHandles(Ts);
                    end
                else
                    if ~isnumeric(Ts) % Format of Ts
                        error("Ts must be a duration or double, not a %s",class(Ts))
                    end
                end
            case 'fs', Fs = value;
                if ~isnumeric(Fs) % Format of Fs
                    error('Fs must be a double, not a %s',class(Fs))
                end
                dt = 1/Fs; % Adminesionnal sampling period
            case 'dj', dj = value;
                if ~isnumeric(dj) % Format of dj
                    error("dj must be a double, not a %s",class(dj))
                end
            case 'permin', Period_Inf = value;
                if ~isnumeric(Period_Inf) % Format of permin
                    error("permin must be a double, not a %s",class(Period_Inf))
                end
            case 'fmin', Freq_Inf = value;
                if ~isnumeric(Freq_Inf) % Format of fmin
                    error("fmin must be a double, not a %s",class(Freq_Inf))
                end
            case 'smin', s0 = value;
                if ~isnumeric(s0) % Format of s0
                    error("s0 must be a double, not a %s",class(s0))
                end
            case 'permax', Period_Sup = value;
                if ~isnumeric(Period_Sup) % Format of permax
                    error("permax must be a double, not a %s",class(Period_Sup))
                end
            case 'fmax', Freq_Sup = value;
                if ~isnumeric(Freq_Sup) % Format of fmax
                    error("fmax must be a double, not a %s",class(Freq_Sup))
                end
            case 'smax', s_max = value;
                if ~isnumeric(s_max) % Format of smax
                    error("smax must be a double, not a %s",class(s_max))
                end
            case 'norm', norm = char(string(upper(value)));
                if length(norm)==1 % For norms without (L)
                    norm=strcat('L',norm);
                end
                if ~ismember(norm,["L1","L2"]) % Value verification
                    error('Normalization method "%s" inadmissible. \nPlease select an option among L1 or L2',norm)
                end
            case 'pad', pad = char(string(lower(value)));
                if ~ismember(pad,{'zpd','rfl','rfle','per','none'}) % Value verification
                    error('Padding method "%s" inadmissible. /nPlease refer to the top commented section of the function',pad)
                end
            case 'padmode', padmode = char(string(lower(value)));
                padmode = padmode(1);
                if ~ismember(padmode,{'r','b','l'}) % Value verification
                    error('Padding direction "%s" inadmissible. \nPlease select an option among right, left or both',padmode)
                end
            case 's2', s2 = value;
                if ~isnumeric(s2) % Format of s2
                    error("s2 must be a double, not a %s",class(s2))
                end
            case 'w0', w0 = value;
                if ~isnumeric(w0) % Format of w0
                    error("w0 must be a double, not a %s",class(w0))
                elseif w0<6 % Value verification
                    warning('The Morlet wavelet is admissible for w0>6. For lower values, additionnal terms cannot be neglected')
                end
            case 'percentout', pctout = value;
                if ~isnumeric(pctout) % Format of pctout
                    error("percentout must be a double, not a %s",class(pctout))
                elseif pctout>=1
                    error('percentout value %s inadmissible. \nPlease specify a value inferior to 1',string(pctout))
                end
            case 'coimethod', coimethod = char(string(lower(value)));
                if ~ismember(coimethod,{'anal','num'}) % Format of coimethod
                    error("COI method '%s' inadmissible. /nPlease use either 'anal' or 'num'",coimethod)
                end
            case 'plot', i = i-1; Figure = true;
            case 'nosig', i=i-1; signif = false;
            case 'time', time = value;
            case 'plotunits', Units = char(string(lower(value)));
                if strcmp(Units(end),'s') % Remove plural form
                    Units = Units(1:end-1);
                end
                if ~ismember(Units,{'sec','min','hour','day','year'}) % Conversion
                    error('Plot units "%s" inadmissible. \nPlease select an option among sec, min, hour, day or year',Units)
                end
            otherwise
                error('Error: Input Argument "%s" inadmissible. \nPlease refer to the top commented section of the function',name);
        end
    end
end



% Exclusive arguments
msg=[];
if any(ismember(DataOut,["Ts" "Fs" "dt" "dtf"]))
    if ~isempty(Ts) && ~isempty(Fs) % Sampling period and frequency
        msg =[msg "Sampling period (Ts) and frequency (Fs)"];
    end
    if isempty(Ts) && isempty(Fs) % No input
        Ts = 1;
        dt = 1; % Adimensionnal temporal resolution
        disp ('Temporal resolution fixed to 1 (adimensionnal)')
    end
end
if any(ismember(DataOut,["Period_Inf" "s0" "Freq_Sup"]))
    if sum([~isempty(Period_Inf), ~isempty(s0), ~isempty(Freq_Sup)])>1 % Minimum period and scale
        msg =[msg "Min period (permin), min scale (smin) and max frequency (fmax)"];
    end
    if ~isempty(Freq_Sup) % fmax specified
        Period_Inf = 1/Freq_Sup; % Converison to period
    end
end
if any(ismember(DataOut,["Period_Sup" "s_max" "Freq_Inf"]))
    if sum([~isempty(Period_Sup), ~isempty(s_max), ~isempty(Freq_Inf)])>1 % Maximum period and scale
        msg =[msg "Max period (permax), max scale (smax) and min frequency (fmin)"];
    end
    if ~isempty(Freq_Inf) % fmin specified    
        Period_Sup = 1/Freq_Inf; % Converison to period
    end
end
if~isempty(msg)
    error('Mutually exclusive arguments specified. Please indicate only one of the following arguments: \n%s',strjoin(msg," ; "))
end

if any(ismember(DataOut,"norm"))
    % Normalisation (value)
    if isempty(norm)
        if ftype(1:3)=="GAU" % Gauss
            norm = 'L2';
        else % Other wavelets
            norm = 'L1';
        end
    end
    % Gaussian function and norm L1
    if strcmp(norm,'L1') && strcmp(ftype(1:3),'GAU')
        warning("Fourier Factor (period to scale conversion) undefined for norm 'L1' using the Gaussian function. Consider setting the norm to 'L2' for correct period conversions.")
    end
end

% Outputs
varsout = {ftype,Ts,dt,dtf,Fs,dj,Period_Inf,s0,Period_Sup,s_max,norm,pad,...
    padmode,s2,w0,pctout,coimethod,Figure,signif,time,Units};
namesout = {'ftype','Ts','dt','dtf','Fs','dj','Period_Inf','s0','Period_Sup','s_max','norm','pad',...
    'padmode','s2','w0','percentout','coimethod','Figure','signif','time','Units'};
varargout = varsout(ismember(namesout,DataOut));
