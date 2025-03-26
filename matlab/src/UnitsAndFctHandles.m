function [dt,fct,tsformat] = UnitsAndFctHandles(Ts)
% Find the dimensionless sampling interval, define a function handle and
% extract units of Ts

tsformat = Ts.Format; % Extract format

% Extract first character of Format
if iscalendarduration(Ts) % If calendar duration used
    warning('It is best to use duration instead of calendarduration for the timestep to avoid ambiguity')
    [y,mo,d,Ts]=split(Ts,{'years','months','days','time'});
    if Ts>0
        % Convert to Hours,Minutes,Seconds
        [h,m,s] = hms(Ts);
        % Find the biggest unit
        timeidx = find([h m s],1,'first');
        switch timeidx
            case 1
                if h>=24
                    tsformat = 'd';
                else
                    tsformat = 'h';
                end
            case 2
                tsformat = 'm';
            case 3
                tsformat = 's';
            
        end
    end
    if d>0
        tsformat = 'd';
        Ts=Ts+days(d);
    end
    Ts=Ts+days(30*mo);
    if y>0
        tsformat = 'y';
        Ts=Ts+years(y);
    end
else
if strcmpi(tsformat,'hh:mm:ss') || strcmpi(tsformat,'dd:hh:mm:ss') ...
        || strcmpi(tsformat,'mm:ss') || strcmpi(tsformat,'hh:mm')
    % Convert to Hours,Minutes,Seconds
    [h,m,s] = hms(Ts);
    % Find the biggest unit
    timeidx = find([h m s],1,'first');
    switch timeidx
        case 1
            if h>=24
                tsformat = 'd';
            else
                tsformat = 'h';
            end
        case 2
            tsformat = 'm';
        case 3
            tsformat = 's';
        
    end
else
    tsformat = tsformat(1);
end
end

% Define dt and function handle
switch tsformat
    case 's'
        dt = seconds(Ts);
        fct = @(x)seconds(x);
        tsformat = 'sec';
    case 'm'
        dt = minutes(Ts);
        fct = @(x)minutes(x);
        tsformat = 'min';
    case 'h'
        dt = hours(Ts);
        fct = @(x)hours(x);
        tsformat = 'hour';
    case 'd'
        dt = days(Ts);
        fct = @(x)days(x);
        tsformat = 'day';
    case 'y'
        dt = years(Ts);
        fct = @(x)years(x);
        tsformat = 'year';
    otherwise
        error('Incorrect duration specified for %s. /nPlease use seconds, minutes, hours, days, or years.','Ts');
end