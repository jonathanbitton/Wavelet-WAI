function [CoefCOI,Coef,idx,Sgn,SgnCOI] = coefvalues(wave,power,periods,COI,Ts,L,varargin)
% Extracting mean periodic bands.

%%% Inputs
% Optional arguments (in order):
% 1) Periods to analyze (string vector), with a letter indicating the units 
% ("y","m","w","d","h" for years, months, weeks, days and hours) followed 
% by the number of time units desired
% Example: "m6" for 6 months. 
% 2) Number of days corresponding to periods (double vector)
% 3) Half-spread (half the number of days in the interval) to compute mean (double vector)

% Options 2) and 3) need both to be entered or omitted
% For details on other entries, see 'cwtransform' function

%%% Outputs
% CoefCOI: Mean power values without border-affected values (COI)
% Coef: Mean power values
% idx: Indexes of coefficients considered for computing mean periodic bands
% SgnCOI: Mean wave values without border-affected values (COI)
% Sgn: Mean wave values


% Defining periods to analyze
if isempty(varargin) % Default
    pnames = ["y1","m8","m7","m6","m5","m4","m3","m2","m1","w3","w2","w1","d1","h12"]; % Period identifier
    pdate = [365 240 210 180 150 120 90 60 30 21 14 7 1 0.5]; % Date to analyze
    pspread = [30 15 15 15 15 15 15 15 15 3.5 3.5 3.5 0.3 0.1]; % Spread for mean computation
    nper = numel(pnames); % Number of periods to extract
else % User-defined periods
    pnames = varargin{1};
    pnames = pnames(:)';
    if numel(varargin)>1 % User-defined dates and spreads
        pdate = varargin{2};
        pspread = varargin{3};
        pdate = pdate(:)';
        pspread = pspread(:)';
    else % Automatic dates and spreads corresponding to periods
        nper = numel(pnames);
        pdate = nan(1,nper);
        pspread = nan(1,nper);
        for j = 1:nper
            i = char(pnames(j));
            if length(i)<2
                error('Period %s not recognized. \nPlease use a letter indicating the period studied ("y","m","w","d","h") followed by the number of time units. \nExample: "y1" for 1 year',string(i))
            end
            no_i = str2double(i(2:end));
            switch i(1)
                case 'y', pdate(j) = no_i*365; pspread(j) = 30;
                case 'm', pdate(j) = no_i*30; pspread(j) = 15;
                case 'w', pdate(j) = no_i*7; pspread(j) = 3.5;
                    if no_i>3 % Convert spread to months
                        pspread(j) = 15; 
                    end
                case d', pdate(j) = no_i; pspread(j) = 0.3; 
                    if no_i>=30 % Convert spread to months
                        pspread(j) = 15; 
                    elseif no_i>=7 % Convert spread to weeks
                        pspread(j) = 3.5; 
                    end
                case 'h', pdate(j) = no_i/24; pspread(j) = 0.1;
                    if no_i>=720 % Convert spread to months
                        pspread(j) = 15; 
                    elseif no_i>=168 % Convert spread to weeks
                        pspread(j) = 3.5; 
                    elseif no_i>=24 % Convert spread to days
                        pspread(j) = 0.3; 
                    end
                otherwise
                    error('Period %s not recognized. \nPlease choose among "y","m","w","d" and "h" for the desired period',string(i))
            end
        end
    end
end

% Indexes of coefficients to consider for computing means
% Initialization
idx=table('Size',[size(power,1),nper],'variableTypes',...
    repmat("logical",1,nper),'VariableNames',pnames);
% Defining coefficients to keep for each period
for i=1:nper
    v1 = pdate(i)+pspread(i);
    v2 = pdate(i)-pspread(i);
    idx.(pnames(i)) = days(periods)<v1 & days(periods)>v2;
end

% Mean computation
% Initialization (see last 4 lines for definitions)
Coef=table('Size',[size(power,2),nper],'variableTypes',...
    repmat("double",1,nper),'VariableNames',pnames); % Coefficients
CoefCOI=Coef;
Sgn=Coef;
SgnCOI=Coef;

% Defining edge-affected coefficients (using COI)
edgeaff = (COI-periods);
edgeaff(edgeaff<=0) = 10e6;
[~,b] = min(edgeaff);
b(COI-2*Ts<1e-8)=0;
edgeaff = ones(size(power,1),size(power,2));
for i = 1:ceil(L/2)
    if b(i)>0, edgeaff(1:b(i),i) = 0; end
end
if rem(L,2)  % Odd data points
   edgeaff = [edgeaff(:,1:ceil(L/2)) fliplr(edgeaff(:,1:fix(L/2)))];
elseif ~rem(L,2)  % Even data point
   edgeaff = [edgeaff(:,1:L/2) fliplr(edgeaff(:,1:L/2))];
end
power2=power;
power2(edgeaff==1)=nan;
wave2=wave;
wave2(edgeaff==1)=nan;

% Compute means
for i=pnames
    Coef.(i) = mean(power(idx.(i),:))'; % Mean power values
    Sgn.(i) = mean(wave(idx.(i),:))'; % Mean wave values
    CoefCOI.(i) = mean(power2(idx.(i),:))';% Mean power values without border-affected values (COI)
    SgnCOI.(i) = mean(wave2(idx.(i),:))';% Mean wave values without border-affected values (COI)
end