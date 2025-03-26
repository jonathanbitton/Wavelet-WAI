function PeakOut = DatesFromCenterToExtremumOrZeroDOG(PeakIn,ftype,pers,Ts,varargin)
% Moves peak position to next maximum or zero of the DOG wavelet
%%% Inputs
% - 'PeakIn' (table): Peaks from 'ExtractPeaks' function
% - 'ftype' (char): Analyzing function 'DOGx'
% - 'pers' (char): Period
% - 'Ts' (duration): Sampling period
% Optional arguments using the format (...,'Argument',Value,'Argument',Value,...):
% 1) 'norm' (char): 'L1' or 'L2'
% 2) 's2' (double): Variance of the Gaussian function

if nargin>8
    error('Too many input arguments')
end

% Extract Fourier Factor and verify inputs
Fact_Fourier = DefFourierF(ftype,varargin{:});

% Formats and Values
if ~ischar(pers) && ~isstring(pers)
    error("pers must be a char or string, not a%s",class(pers))
end
if ~strcmp(ftype(1:3),'DOG') || sscanf(ftype,'DOG%d')==0
    error('Wavelet "%s" inadmissible. \nPlease enter a DOGx (x=order of derivation >1) wavelet',ftype)
end
if ~isduration(Ts)
    error("Ts must be a duration, not a%s",class(Ts))
end

% Extract desired peaks (wavelet position)
PeakOut = PeakIn(:,contains(PeakIn.Properties.VariableNames,pers));
% Initialize
PeakOut{:,contains(PeakOut.Properties.VariableNames,'Value')} = nan;

% Moive wavelet to next Peak/Zero (add a/Fourier_Factor)
for i =1:length(pers)
    % Extract period in days
    pers2 = char(pers(i));
    no_pers = str2double(pers2(2:end));
    switch pers2(1)
        case 'y', ai = years(no_pers);
        case 'm', ai = days(no_pers*30);
        case 's', ai = days(no_pers*7);
        case 'd', ai = days(no_pers);
        case 'h', ai = days(no_pers/24);
    end
    ai.Format = 'd';
    % Translation
    bi = ai/Fact_Fourier; % Translation in days
    bi_rem = rem(bi,Ts); % Extract non integer multiples of Ts
    % Round to integer multiples of sampling period 
    if bi_rem<Ts/2
        addvalue = bi-bi_rem; % round "down"
    else
        addvalue = bi-bi_rem+Ts; % round "up"
    end
    % Move peak dates
    PeakOut.(strcat(pers2,"Date"))=PeakOut.(strcat(pers2,"Date"))+addvalue;
end