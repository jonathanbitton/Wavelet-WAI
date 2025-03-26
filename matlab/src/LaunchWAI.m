function LaunchWAI(date,dtb,ftype,Months,var,time, ...
    Ts,wbk,mode,Title,PeakRep,SPeaks,Peaks,Per,varargin)
% Plot computation steps for wavelet transform (WAI)

%%% Input arguments :
% 'date' (datetime or double): Date of peak - years to consider for peak filtering
% 'dtb' (string): Period to consider
% 'ftype' (char): Analyzing function used among 'Morlet', 'Mexhat', 'DOGx', 'Gauss', 'Haar'
% 'Months' (double): Months to consider for peak filtering
% 'var' (double): Signal
% 'time' (datetime): Timestamps of signal
% 'Ts' (duration): Sampling period
% 'wbk' (struct): Structure array with results of 'cwtransform'
% 'mode' (char): 'together' to show peaks spanning  on the same plot 
% 'Title' (char): Title. Set to 'default' to display period (+ time) on plot
% 'PeakRep' (struct): Repeating peaks, fields = designation of repeated peaks
% 'SPeaks' (struct): Filetered peaks, fields = periods (such as dtb2)
% 'Peaks' (struct): All peaks, fields = periods (such as dtb2)
% 'Per' (char): Peaks to study: All ('All'), filtered ('Sup') or Repeated 
% peaks (same names as fields in PeakRep, e.g. 'Apr', 'Jun' or 'Nov')
% Optional arguments using the format (...,'Argument',Value,'Argument',Value):
% 1) 's2' (double): Variance of the Gaussian function used to define wavelets
% 2) 'w0' (double): Central angular frequency of the Morlet wavelet (ignored if ftype not set to Morlet)

% If 'date' is a datetime, 'Per', 'Months' and all peak data structures are
% ignored
% If 'Per' is set to a repeated peak, 'Months' is ignored

% Initialize
% Convert Inputs and verify format/values
[ftype,dt,PeakRep,SPeaks,Peaks,dtb,ai,mode,Title,periods,scales,s2,w0] = ...
    VerifyConvertInputs(date,ftype,dtb,Months,wbk,Ts,PeakRep,SPeaks,Peaks,Per,mode,Title,varargin{:});

% Finding peaks
if isdatetime(date) % Date(s) specified
    PksData = date;
    PksData(ismissing(PksData),:)=[]; % Get rid of NaNs
else % Peaks in structures
    % Peak type and period selection
    if strcmpi(Per,"Sup") % Filtered peaks
        PksData = SPeaks.(dtb); % Extract filtered peaks and their location
    elseif strcmpi(Per,"All") % All peaks
        PksData = Peaks.(dtb); % Extract all peaks and their location
    else  % Repeated peaks
        if ~ismember(fields(PeakRep),Per)
            error("No field %s found inside 'PeakRep' for ftype. \nValue entered for 'Per' must match a field of 'PeakRep' for ftype",string(Per))
        end
        indices = arrayfun(@(x) ~isempty(strfind(x{1}, dtb)), PeakRep.(Per).Properties.VariableNames); % Find columns with peaks for desired period
        PksData = PeakRep.(Per)(:,indices); % Extract repeated peaks and their location
        PksData(isnan(PksData{:,2}),:)=[]; % Get rid of NaNs
    end
    % Date Selection (filter on years and/or months)
    PksData = PksData(ismember(year(PksData{:,1}),date),:); % Keep only desired years
    if ismember(Per,["Sup" "All"]) % If Per set to repeated peaks, Months ignored
        PksData = PksData(ismember(month(PksData{:,1}),Months),:); % Keep only desired months
    end
    % Check if peaks found
    if isempty(PksData)
        error("No peaks found for %s",dtb)
    end
    PksData = PksData{:,1}; % Extract dates
end

% Graphs together or separately
if length(PksData)==length(unique(year(PksData))) && strcmp(mode(1),'t') && height(PksData)>1 % Multiple wavelets on figure
    % Defining wavelets/functions and product curves/areas
    for i = 1:length(PksData)
        bi = PksData(i);
        temp = ComputeWave(var,time,periods,scales,dt,ai,bi,ftype,s2,w0);
        if i==1
            Tb = temp;
        else
            j = find(abs(temp.(ftype))-abs(Tb.(ftype))>1e-5,1,'first');
            Tb{j:end,:} = temp{j:end,:};
        end
    end
    % Title
    if strcmpi(Title,'default')
        Title = strcat("Period : ", string(ai));
    end
    % Plot : wavelet coefficient computation steps
    PlotWAI(Tb,ftype,Title)
else
    for i=1:height(PksData)
        bi = PksData(i);
        Tb = ComputeWave(var,time,periods,scales,dt,ai,bi,ftype,s2,w0);
        if strcmpi(Title,'default')
            Title2 = strcat("Period : ", string(ai)," - Time : ", string(bi));
		else
			Title2 = '';
        end
        % Plot
        PlotWAI(Tb,ftype,Title2)
    end
end




function [ftype,dt,PeakRep,SPeaks,Peaks,dtb,ai,mode,Title,periods,scales,s2,w0] = ...
    VerifyConvertInputs(date,ftype,dtb2,Months,wbk,Ts,PeakRep,SPeaks,Peaks,Per,mode,Title,varargin)

% Structures verification
if ~isdatetime(date)
    if strcmpi(Per,"Sup")
        if ~isstruct(SPeaks),error('SPeaks structure not defined');end
        if ~ismember(ftype,fields(SPeaks))
            error('SPeaks structure not defined for %s',ftype)
        end
        SPeaks = SPeaks.(ftype);
    elseif strcmpi(Per,"All")
        if ~isstruct(Peaks),error('Peaks structure not defined');end
        if ~ismember(ftype,fields(Peaks))
            error('Peak structure not defined for %s',ftype)
        end
        Peaks = Peaks.(ftype);
    else
        if ~isstruct(PeakRep),error('PeakRep structure not defined');end
        if ~ismember(ftype,fields(PeakRep))
            error('PeakRep structure not defined for %s',ftype)
        end
        PeakRep = PeakRep.(ftype);
    end
end

% Ts format
if ~isduration(Ts) 
    error("Ts must be a duration, not a %s",class(Ts))
end
[dt,fct] = UnitsAndFctHandles(Ts);

% Extract periods/scales analyzed
if ~isstruct(wbk)
    error("Wavelet structure 'wbk' empty")
end
if ismember(ftype,fieldnames(wbk))
    periods = wbk.(ftype).periods;
    scales = wbk.(ftype).scales;
    % Wavelet (ftype) format & parameters
    [ftype,s2,w0]=GetArgs({'ftype','s2','w0'},'ftype',ftype,varargin{:});
else % Need to define scales
    nameF = fieldnames(wbk);
    nameF = nameF{1};
    warning("wbk structure not defined for '%s'. \nUsing periods computed for '%s' and converting to scales using default parameters (wavelet and norm). \nTo modify these, please launch 'cwtransform' with '%s' and store outputs in 'wbk'",ftype,nameF,ftype)
    periods = wbk.(nameF).periods;
    % Fourier Factor, wavelet (ftype) format & parameters
    [Fact_Fourier,ftype,s2,w0] = DefFourierF(ftype,varargin{:});
    scales = fct(periods)/Fact_Fourier;
end

% Months value and format
if ~isnumeric(Months)
    error("Months must be a double, not a %s",class(Months))
end
if any(Months>12)
    error("Months cannot exceed 12")
end

% mode format
mode = char(mode);

% Title format
Title = char(Title);

% Convert period
dtb = char(dtb2);
no_dtb = str2double(dtb(2:end));
switch dtb(1)
    case 'y', ai = years(no_dtb);
    case 'm', ai = days(no_dtb*30);
    case 's', ai = days(no_dtb*7);
    case 'd', ai = days(no_dtb);
    case 'h', ai = days(no_dtb/24);
    otherwise
        error("Period to consider %s indamissible. It must consist of a letter ('y','m','s','d','h') followed by a number",dtb)
end

