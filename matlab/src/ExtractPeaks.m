function [Peak,SPeak] = ExtractPeaks(CoefCOI,time,Sgn,dtb2,mts,varargin)
% Extracting repeating peaks in mean periodic bands for different years

%%% Inputs
% - CoefCOI (double): Mean power values without border-affected values (COI)
% - time (datetime): time corresponding to each signal data point
% - Sgn (double): Mean wave values without border-affected values (COI)
% - dtb2 (char): Period band(s) to extract peaks
% - mts (double): Months to consider for peak extraction
% Optional argument:
% 1) thresh (double): Set a value threshold to filter peaks

%%% Outputs
% Peak: Repeated peaks found
% SPeak: Filtered peaks

if nargin>6,error('Too many input arguments'),end

% Periods
no_dtb = length(dtb2); % Number of periods required
dtb2 = Psort(dtb2); % Sort periods in descending order

% Names of output tables
names = strings(1,2*no_dtb);
names(1:2:end) = dtb2+"Date";
names(2:2:end) = dtb2+"Value";

% Initializing peaks data
years = unique(year(time(:)')); % Periods of study (years)
years = strsplit(num2str(years)); % Convert to string
Peak = table('Size',[length(years), length(names)], ...
    'variableTypes',repmat(["datetime", "double"],1,no_dtb), ...
    'VariableNames',names, ...
    'RowNames',years);
Peak{:,2:2:end}(Peak{:,2:2:end}==0)=nan;
% SPeak

if ~isempty(varargin)
    thresh = varargin{1};
else
    thresh =-Inf;
end
%%
% Loop through all desired periods
for j = 1:no_dtb
    dtb = char(dtb2(j));
    [pk,ts] = FindMax(CoefCOI{:,dtb},time); % Find peaks
    wave = Sgn{ismember(time,ts),dtb}; % mean wavelet coefficients, left for sign filtering (to be added)
    CoefM = table(ts,pk,wave); % Table of peaks
    CoefMAX = CoefM(CoefM.pk>thresh,:); % Threshold to delete peaks
    % Repeating peaks
    idx = CoefMAX(ismember(month(CoefMAX.ts),mts),{'ts','pk'}); % Index of peaks in desired months
    if height(idx)>length(unique(year(idx.ts))) % More than one peak per year
        idx.Year = year(idx.ts);
        for i = year(idx.ts(1)):year(idx.ts(end))
            date = idx.ts(idx.Year==i);
            if length(date)>1    
                dtprev = strcat(dtb(1),num2str(str2double(dtb(2:end))+1));
                if ismember(strcat(dtprev,'Date'),Peak.Properties.VariableNames)
                    [~,b] = min(abs(date-Peak{string(i),strcat(dtprev,'Date')}));
                    idx(idx.Year==i & idx.ts~=date(b),:) = [];
                else
                    warning(['Unable to filter peaks for year %s starting %s. ' ...
                        '\nAs an alternative solution, highest peak values will be retained for %s. ' ...
                        '\nPlease add higher periods (%s) to enable tracking peaks in lower periods'],string(i),dtb,dtb,dtprev)
                    [~,b] = max(idx.pk(idx.Year==i));
                    idx(idx.Year==i & idx.ts~=date(b),:) = [];
                end
            end
        end
    end
    Peak(string(year(idx.ts)),[j*2-1,j*2]) = idx(:,1:2);
    % Filtered peaks
    SPeaks2 = CoefM(~ismember(CoefM.ts,idx.ts),:);
    SPeak.(dtb) = SPeaks2;
end
%%

function [yP,xP] = FindMax(y,x)
% Remove repeating values
i = 1:length(y);
i = i([true;(diff(y(:))~=0)]);

% Find indices of local maxima for non repeating values
u = sign(diff(y(i))); % Sign of first derivative
u2 = diff(u); % Difference of signs
idx = find(u2 == -2); % Local Maxima when negtaive sign => u2=-2
idx = idx  + 1; % Adding first value (omitted by diff)

% Translate indices of local maxima for initial vecor y
idx = i(idx); % Final indices for maxima

% Extract y and x values of maxima
yP = y(idx);
xP = x(idx);


function dtb3 = Psort(dtb2)
% Sort periods (years => hours)

% Initializing
dtb3 = strings(1, length(dtb2));
k=1;

% Sorting
for persT = ["y" "m" "w" "d" "h"]
    pdtest = ~ismissing(regexp(dtb2(:), strcat('^',persT,'\d+$'), 'match','once')); % Find matches
    if any(pdtest)
        dtbtemp = dtb2(pdtest); % Elements found
        l = length(dtbtemp); % Number of findings
        dtbtemp = regexp(dtbtemp(:), '\d+', 'match','once'); % Extract numbers
        dtbtemp = sort(str2double(dtbtemp),'descend'); % Sort
        dtb3(k:k+l-1) = persT+string(dtbtemp); % Insert in final vector
        k = k+l; % Increment index
    end
end