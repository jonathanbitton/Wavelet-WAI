function plotscalogram(wave,fper,time,coi,TFs,varargin)
% Plotting scalogram (code inspired from the cwt function)

%%% Inputs:
% - wt (double): Wavelet coefficients
% - fper (double or duration): Periods studied or frequencies (same units as TFs) 
% - time (double or duration): Time vector on the x-axis of the scalogram
% - coi (double or duration): Cone of influence (same units as TFs) 
% - TFs (double or duration): Sampling period or Frenquency
% Optional arguments using the format (...,'Argument',Value,'Argument',Value,...):
% 1) 'FP' (char): 'frequency' or 'period', matching fper
% 2) 'signif' (double):  Significance of wavelet coefficients (Values > 1 
% indicate significant coefficients)
% 3) 'plotunits' (char): Display units for graph periods among 'sec','min',
% 'hour','day','year'

% Default values for optional arguments:
% 1) 'FP': period
% 2) 'signif': no significant coefficients
% 3) 'plotunits': same units as TFs

%%% Outputs : 
% Scalogram plot

%%% Example: 
% n = length(y);
% Ts = minutes(30); % En minutes
% time = 0:minutes(Ts):n*minutes(Ts)-minutes(Ts); % En minutes
% [wt,~,period,~,coi,signif] = cwavtransform(y,'Ts',Ts);
% plotscalogramp(wt,period,time,coi,Ts,'signif',signif,'plotunits','hours') % Plot en heures sur une année

if nargin>11
    error('Too many input arguments')
end

% Import arguments
[signif,Units2,FP] = getargplot(varargin{:});


% Matlab version
v = version('-release');
year = str2double(v(1:end-1));

% Units for periods & COI
if year>2014 || (year==2014 && v(end)=='b')
    if any(xor(isduration(TFs),[isduration(coi),isduration(fper)]))
        error('COI, periods and Ts must have matching formats (duration or double)')
    elseif isduration(TFs)
        [~,fct,Units] = UnitsAndFctHandles(TFs);
        if ~isempty(Units2) % Unit conversion
            if ~strcmp(Units,Units2) % If Ts unit does not match plotunits
                [Units,fct] = convertUnits(Units2);
            end
        end 
        % Dimensionless periods/frequencies
        fper = fct(fper);
        % Dimensionless COI
        coi = fct(coi);
    end
end

% Time vector
if year>2014 || (year==2014 && v(end)=='b')
    if isdatetime(time)
        t = datenum(time);
        X = 1;
    else
        t = time;
        X = 2;
    end
else
    t = time;
    X = 2;
end

%% Plot
f = figure;
AX = axes('parent',f);
% Plotting the scalogram
if strcmp(FP,'p')
    Ypers = [min(fper) max(fper)];
    ylbl = 'Period';
else
    Ypers = [max(fper) min(fper)];
    ylbl = 'Frequency';
end

surface('Parent',AX,...
    'XData',[min(t) max(t)],'YData',Ypers,...
    'CData',abs(wave), 'ZData', zeros(2,2), ...
    'CDataMapping','scaled', 'FaceColor','texturemap', 'EdgeColor', 'none');

% Modifying limits and scales of axes

set(AX,...
    'YLim', [min(fper), max(fper)],...
    'XLim', [min(t), max(t)],...
    'Layer', 'top',...
    'YScale', 'log')

% Labels and ticks
title(AX,'Scalogram');
if exist('Units','var')
    ylabel(AX,strcat(ylbl," (",Units,')'))
else
    ylabel(AX,ylbl)
end
% Modify X tick labels
if X==1 
    datetick(AX,'x','keeplimits');
elseif exist('Units','var') 
    xlabel(AX,strcat('Time (',Units,')')); 
else
    xlabel(AX,strcat('Time')); 
end
% Modify Y tick labels
set(AX,'YTickLabel',get(AX,'YTick'));

% Colorbar and limits
cbar = colorbar(AX);
set(get(cbar,'Label'),'String','Amplitude');
lims = caxis;
caxis([lims(1),lims(2)])

hold(AX,'on');
% Cone of Influence
plot(AX,t,coi,'w--','linewidth',2,'PickableParts','none');
COIaffected = area(AX,t,coi,Ypers(2),'PickableParts','none');
set(COIaffected,'EdgeColor','none','FaceColor',[0.5 0.5 0.5]);
alpha(COIaffected,0.6);

% Significance
if ~isempty(signif)
    contour(AX,t,fper,signif,[1,1],'k','LineWidth',0.5,'PickableParts','none'); 
end

% Data cursor
set(datacursormode(f),'SnapToDataVertex','off')



function [NewUnits,Newfct] = convertUnits(Units)

Units = lower(Units(1));

switch Units
    case 's'
        Newfct = @(x)seconds(x);
        NewUnits = 'sec';
    case 'm'
        Newfct = @(x)minutes(x);
        NewUnits = 'min';
    case 'h'
        Newfct = @(x)hours(x);
        NewUnits = 'hour';
    case 'd'
        Newfct = @(x)days(x);
        NewUnits = 'day';
    case 'y'
        Newfct = @(x)years(x);
        NewUnits = 'year';
    otherwise
        error('Incorrect duration specified for %s. /nPlease use sec, min, hour, day, or year.','Ts');
end

function [signif,Unitsplot,FP] = getargplot(varargin)
signif = [];
Unitsplot = [];
FP = 'p';

% Optional inputs
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
            case 'signif', signif = value;
            case 'plotunits', Unitsplot = char(string(lower(value)));
            case 'fp', FP = char(string(lower(value)));
            otherwise
                error('Error: Input Argument "%s" inadmissible. \nPlease refer to the top commented section of the function',name);
        end
    end
end

% Plotting units format and values
if ~isempty(Unitsplot)
    if strcmp(Unitsplot(end),'s')
        Unitsplot = Unitsplot(1:end-1); % Remove plural
    end
    if ~ismember(Unitsplot,{'sec','min','hour','day','year'})
        error('Plot units "%s" inadmissible. \nPlease select an option among sec, min, hour, day or year',Unitsplot)
    end
end

% signif format
if ~isnumeric(signif)
    error('signif must be a double, not a %s',class(signif))
end

% FP format
if ~ismember(FP(1),{'f','p'})
    error('FP value "%s" inadmissible. \n Please select an option among frequency or period',FP)
end
FP = FP(1);