% Example script: using WAI and producing indicators & phenological metrics

% Ensure setup is run before using functions

% Get script directory
scriptDir = fileparts(mfilename('fullpath'));

% Run setup.m to configure paths
run(fullfile(scriptDir, '..', 'setup.m'));

% Load DPP data and time vector
[GPP,TimeStr] = LoadData(fullfile(scriptDir, '..', 'data', 'GPPdata.txt'));

% Initializing variables to save results
wbk = struct([]); % For GPP signal
wbknorm =  struct([]); % For Normalized GPP signal

% Sample collection details
dt = 30; % Time interval between samples (30 minutes), unitless
time = datetime(TimeStr); % Conversion to datetime
Ts = minutes(dt); % Sampling period

% Parameters of CWT
dj = 1/20;

%% Step 1a: Scalogam
% Compute the CWT and plot scalogram
cwtransform(GPP,'Ts',Ts)

%% Step 1b: Compute mean periodic bands
ftype = 'DOG2'; % Wavelet to use
per = ["y1" "m"+(6:-1:1)]; % Periods to consider: 1 year and 1-6 months
% Format: letter indicating the units ("y","m","w","d","h" for years, months, 
% weeks, days and hours) followed by the number of time units desired.

% Compute mean periodic bands (usig "cwtransform" and "coefvalues")
wbk = ComputeMeanPeriodicBands(per,wbk,GPP,ftype,Ts); % function below script

%% Step 1c: Plot mean periodic bands
per= "m6"; % Period to represent (e.g. 6 months)
ftype = 'DOG2'; % Wavelet to use (after running step 1b)

% Plot mean periodic bands for period "per"
figure
plot(time,wbk.(ftype).Coef{:,per},'LineStyle','--','HandleVisibility','off', ...
    'Color',[0.00,0.45,0.74]) % Edge-affected coefficients
hold on
plot(time,wbk.(ftype).CoefCOI{:,per},'LineStyle','-', ...
    'Color',[0.00,0.45,0.74]) % Unaffected coefficients
% Titles and labels
title(strcat("Mean periodic band (",per,")"))
ylabel(strcat("Mean wavelet coefficient power (",ftype,")"))
% Found repeating peaks in March/April, June and November

%% Step 1d: Select peaks in periodic bands
% Recommanded to manually check the peak filtering result (PeakRep) and vizualize them (Step 2)

dtbAll = "m"+string(6:-1:2); % Periods to study (e.g. 6 months to 2 months)
ftype = 'DOG2'; % Wavelet to use (after running step 1b)

% April peak
mts = [3 4]; % Months to keep for peak detection (to be adjusted for different datasets)
[PeakApr,SPeakApr] = ExtractPeaks(wbk.(ftype).CoefCOI,time,wbk.(ftype).Sgn,dtbAll,mts);

% June peak
mts = [5 6 7];
[PeakJun,SPeakJun] = ExtractPeaks(wbk.(ftype).CoefCOI,time,wbk.(ftype).Sgn,dtbAll,mts);

% November peak
mts = [10 11];
[PeakNov,SPeakNov] = ExtractPeaks(wbk.(ftype).CoefCOI,time,wbk.(ftype).Sgn,dtbAll,mts);

% Store repeating peaks
PeakRep.(ftype) = struct('Apr',PeakApr,'Jun',PeakJun,'Nov',PeakNov);
% Store all peaks (Peaks) and filtered peaks (SPeaks, isolated / non repeating peaks)
[Peaks.(ftype),SPeaks.(ftype)] = FilteredPeaks(dtbAll,PeakApr,SPeakApr,PeakJun,SPeakJun,PeakNov,SPeakNov);

clear PeakApr PeakJun PeakNov SPeakApr SPeakJun SPeakNov

%% Step 2: Vizualize wavelet coefficients
% Used to interpret peaks and check if lower peaks correspond to the same event than higher ones

% Localization of the peak(s)
date = 2000:2020; % For all years
% date = [2005 2008]; % For specific year(s)
% date = PeakRep.DOG2.Apr.m2Date; % For peak dates 
% date = datetime({'26-Jun-2000 19:45:00'}); % For a single peak (manual)

% Wavelet to use (Recommended to launch Step 1b with ftype beforehand)
ftype = 'DOG2'; % 'Morlet', 'Mexhat', 'DOGx', 'Gauss' or 'Haar'

% Period(s) to consider
perband = "m2"; % See comment at Step 1b for format to use
% If date is set to a numeric value, only periods of 'dtbAll' are accepted

% Peaks to study: Repeated peaks ('Apr', 'Jun' or 'Nov'), filetered peaks 
% ('Sup') or All peaks ('All'). Ignored if 'date' fixed to a datetime
Perpeak = 'Apr' ; % Repeated peak ('Apr', 'Jun', 'Nov'), 'Sup' or 'All' 

% Months of peaks. Ignored if 'date' fixed to a datetime or if 'Perpeak' 
% set to a repeated peak ('Apr', 'Jun' or 'Nov')
Months = 1:12; % Included months, between 1 and 12 for Jan-Dec

% Plot coefficients together or separetely. Ignored if more than one peak 
% per year
mode = 'together' ; % 'together' for coefficients in the same graph

% Title for plots (char)
Title = 'default'; % Title on plot ('default' to show period)

% If code run for a specific date, not for identified peaks
if ~exist('Peaks','var'), Peaks=1; end
if ~exist('PeakRep','var'), PeakRep=1; end
if ~exist('SPeaks','var'), SPeaks=1; end

% Plotting - WAI
LaunchWAI(date,perband,ftype,Months,GPP,time, ...
    Ts,wbk,mode,Title,PeakRep,SPeaks,Peaks,Perpeak)

%% Step3: Refine & Create Indicators
%% First Indicator : Steepness of GPP Rise
% Wavelet and periods
ftype = 'DOG1'; % Wavelet to use
per = "m"+(7:-1:2); % Periods: start at m7 (for I2) to track peaks until m3 (for I1)

% Normalizing by yearly maximum values
GPPsmooth = movmean(GPP,40*48,'omitnan');
MAXS = retime(timetable(time,GPPsmooth),'yearly','max');
GPPnorm = GPP./MAXS{year(time)-year(time(1))+1,:};

% Mean periodic bands on normalized signal
wbknorm = ComputeMeanPeriodicBands(per,wbknorm,GPPnorm,ftype,Ts,'dj',dj);

% April peaks
mts = [4 5 6]; % Months (later than DOG2)
PeakApr = ExtractPeaks(wbknorm.(ftype).CoefCOI,time,wbknorm.(ftype).Sgn,per,mts);

% Indicator
I1 = PeakApr.m3Value; % 3 months
I1(5) = nan; % Sensor malfunctionning in June 2004 => Delete 2004

% Plotting - WAI
mode = 'together' ;
LaunchWAI(PeakApr.m3Date,"m3",'DOG1',[],GPP,time, ...
    Ts,wbknorm,mode,'I1: DOG1 April peak (90 days - normalized signal)',[],[],[],[])
%% Second Indicator : GPP peaking values
% Wavelet and period selection
per = "m7"; % Period to consider, based on support = ~2*(base to peak GPP time) = 2*60days
disp(strcat("For period selected (",per,"), support = ",string(round(days(SupportWav('Gauss',days(7*30),Ts,'percentout',0.01))))," days (used for I2)"))

% June values as defined by Gaussian positionned on dates of min(DOG1) for each period
PeakAprGDOG1 = DatesFromCenterToExtremumOrZeroDOG(PeakApr,'DOG1',per,Ts); % Find minimum of wavelet

% Gaussian Analyzing function
ftype = 'Gauss';
wbk = ComputeMeanPeriodicBands(per,wbk,GPP,ftype,Ts,'dj',dj); % function below script

% June values by positionning Gaussian on DOG1 times (follows ideal peak + values)
I2 = wbk.(ftype).CoefCOI{ismember(time,PeakAprGDOG1.(strcat(per,"Date"))),per};
I2 = [nan ; I2]; % Add nan for 2000 (missing values at the beginning of the year)
I2(5) = nan; % Delete 2004

% Plotting - WAI
mode = 'together' ;
LaunchWAI(PeakAprGDOG1.(strcat("m7","Date")),"m7",'Gauss',[],GPP,time, ...
    Ts,wbk,mode,'I2: Gauss (right lobe of DOG1 April peak - 210 days)',[],[],[],[])

%% Third Indicator : GPP drop importance
ftype = 'DOG2'; 
I3 = PeakRep.(ftype).Jun.m6Value./PeakRep.(ftype).Nov.m6Value; % Ratio of June and November
I3(5) = nan; % Delete 2004

% Plotting - WAI
mode = 'together' ;
LaunchWAI(date,"m6",'DOG2',Months,GPP,time,Ts,wbk,mode, ...
    'I3 - numerator: DOG2 June peak (180 days)',PeakRep,SPeaks,Peaks,"Jun")
LaunchWAI(date,"m6",'DOG2',Months,GPP,time,Ts,wbk,mode, ...
    'I3 - denominator: DOG2 November peak (180 days)',PeakRep,SPeaks,Peaks,"Nov")
%-----------------
%% Phenology %%%
ftype = 'DOG2';
% Start GPP : Mexican Hat April peak at 2 months
BeginDate = PeakRep.(ftype).Apr.m2Date;

% Drop dates : Second zero of Mexican Hat June peak
per = "m6";
DropDate = DatesFromCenterToExtremumOrZeroDOG(PeakRep.(ftype).Jun,ftype,per,Ts); % Position of peaks
DropDate = DropDate{:,1};

% End dates
EndDate = PeakRep.(ftype).Nov.m2Date;

% Plotting - Dates
figure
hold on
% Creating array in DOY
Pheno = [BeginDate DropDate EndDate];
Pheno = day(Pheno,'dayofyear');
% Durations
plot(Pheno(:,[1 2])',[2000:2020;2000:2020],'Color',[0.67,0.83,0.44],'LineWidth',3,'HandleVisibility','off')
plot(Pheno(:,[2 3])',[2000:2020;2000:2020],'Color',[0.67,0.72,0.60],'LineWidth',3,'HandleVisibility','off')
% Events
scatter(Pheno(:,1), 2000:2020,'square','filled','MarkerFaceColor',[0.70,0.95,0.34])
scatter(Pheno(:,2), 2000:2020,'square','filled','MarkerFaceColor',[0.53,0.72,0.25])
scatter(Pheno(:,3), 2000:2020,'square','filled','MarkerFaceColor',[0.33,0.43,0.19])
% Axes and Legend
set(gca,'YDir','reverse','YTick',2000:2020)
xlabel('Day of year')
legend1 = legend({'Growth Onset','GPP Drop','Growth End'});
set(legend1,'Position',[0.166 0.943 0.712 0.056],'Orientation','horizontal')

%--------------------------------Internal functions--------------------------------%
function [GPP,Time]=LoadData(filePath)
% LoadData loads data from a CSV file located in the "data" folder.
%
% Parameters:
%   filePath - Name of the data file (if in same folder) or complete path
%
% Returns:
%   data - Numeric data from the second column
%   Time - Time or string data from the first column

% Open file
fileID = fopen(filePath, 'r');
if fileID == -1
    error('File "%s" not found.', filePath);
end

% Define Format
formatSpec = '%s%f'; % s for strings and f for double

% Read data
dataArray = textscan(fileID, formatSpec, 'Delimiter', ',', 'HeaderLines', 1);

% Close file
fclose(fileID);

% Store data
Time = dataArray{1};
GPP = dataArray{2};
end


function [Peaks,SPeaks] = FilteredPeaks(dtbAll,Peak1,SPeak1,Peak2,SPeak2,Peak3,SPeak3)
% Group all Peaks and filtered peaks
for dtb=dtbAll
    AllPeaks = unique([SPeak1.(dtb);SPeak2.(dtb);SPeak3.(dtb)]);
    Peaks.(dtb) = AllPeaks;
    SPeaks.(dtb) = AllPeaks(~ismember(AllPeaks.ts,[Peak1{:,strcat(dtb,'Date')};Peak2{:,strcat(dtb,'Date')};Peak3{:,strcat(dtb,'Date')}]),:);
end
end

function wbk = ComputeMeanPeriodicBands(pers,wbk,sig,ftype,Ts,varargin)
% Compute mean periodic bands
% varargin contain entries (same format) for cwtransform, except ftype and Ts

% Continuous Wavelet Transform
if isempty(wbk) || any(~ismember(fields(wbk),ftype)) % If computation not made for ftype
    % Input for CWT
    inputCWT = [{'ftype',ftype,'Ts',Ts} varargin];
    % CWT
    [wave,power,periods,scales,COI] = cwtransform(sig,inputCWT{:});
    % Store
    wbk(1).(ftype).wave = wave;
    wbk.(ftype).power = power;
    wbk.(ftype).periods = periods;
    wbk.(ftype).scales = scales;
    wbk.(ftype).COI = COI;
else % Retrieve
    wave = wbk.(ftype).wave;
    power = wbk.(ftype).power;
    periods = wbk.(ftype).periods;
    COI = wbk.(ftype).COI;
end

% Mean periodic bands
if ~ismember('CoefCOI',fields(wbk.(ftype))) % Not stored yet
    % Compute
    [CoefCOI,Coef,idx,Sgn] = ...
        coefvalues(wave,power,periods,COI,Ts,length(sig),pers);
    % Store
    wbk.(ftype).CoefCOI = CoefCOI;
    wbk.(ftype).Coef = Coef;
    wbk.(ftype).idx = idx;
    wbk.(ftype).Sgn = Sgn;
elseif ~all(ismember(pers,wbk.(ftype).CoefCOI.Properties.VariableNames)) % All periods not stored
    % Keep only unique periods
    pers = pers(~ismember(pers,wbk.(ftype).CoefCOI.Properties.VariableNames));
    % Compute
    [CoefCOI,Coef,idx,Sgn] = ...
        coefvalues(wave,power,periods,COI,Ts,length(sig),pers);
    % Store
    wbk.(ftype).CoefCOI = [CoefCOI,wbk.(ftype).CoefCOI];
    wbk.(ftype).Coef = [Coef,wbk.(ftype).Coef];
    wbk.(ftype).idx = [idx,wbk.(ftype).idx];
    wbk.(ftype).Sgn = [Sgn,wbk.(ftype).Sgn];
end
end

