function COIf = DiracCOIComputation(ftype,n,Ts,scales,varargin)
% Compute COI using Dirac impulses at the edges of the signal

%%% Inputs
% - ftype (char): Analyzing function used among 'Morlet', 'Mexhat', 'DOGx', 'Gauss', 'Haar'
% - n (double): Signal length
% - Ts (duration or double): Sampling period
% - scales (double): scales used for the wavelet transform
% Optional argument:
% 1) threshold (double): percentage of edges effect desired

threshold = 0.02; % qcoi
if nargin==5, threshold = varargin{1}; end
% Dirac impulse (length n+2)
dirac = zeros(n+2,1);
dirac([1 end]) = 1; % 1 after the edges of the signal
% Initialization
pad = 'none';
dj = log2(scales(end)/scales(1))/(length(scales)-1);
% Wavelet coefficients
[wave,~,periodes] = ...
    cwtransform(dirac,'ftype',ftype,'Ts',Ts,'dj',dj,'smin',scales(1),'smax',scales(end),'pad',pad,'raw');

% Normalizing each scale by max(wave)
wave = wave./max(wave,[],2);
% Removing added values at boundaries
wave = wave(:,2:end-1); 

COI = zeros(size(wave,1),size(wave,2)); % COI initialization
COI(abs(wave)>threshold) = 1; % Set affected values to 1

% Correction for affected small scales (COI should only increase)
[NumZeros,I]=sort(sum(COI,2)); % Search for rows with minimum of values affected and sort result
% Ligne minimum = on ne trouve pas de coefficient affecté ailleurs que le début et fin de la ligne
for i = 1:length(I) % Search for the line i starting with a minimum of "1" followed only by "0"
    if sum(COI(I(i),NumZeros(i)/2+1:n/2))==0 % Symetrical wavelets : Same number of values out of COI for left and right
        break
    end
end
limit_L = I(i)-1; % Lines to set equal to 0 (lower than limit_L)
limit_C = NumZeros(i)/2; % Corresponding columns (starting limit_C)

if limit_L>0, COI(1:limit_L,limit_C:end-limit_C) = 0; end % Remove affected lower values
% Keep a single unit value per column to define COI limits
COI = cumsum(COI);
COI(COI>1)=0;
ncorr = n-sum((sum(COI)==0)); % Remove columns with unaffected coefficients for all scales

[a,b] = find(COI==1); % COI limits

% Deleting repeating unit values, if "breakthroughs" in the cone (0 under a
% value of 1)
k=0;
if length(b)~=ncorr
    for i = 2:length(b)
        if b(i-1-k)==b(i-k)
            b(i-k) = [];
            a(i-k) = [];
            k = k+1;
        end
    end
end
L = length(b);

if ncorr~=L, error('Longueur du signal non constante'), end

% COI values with units
if ncorr<n
    COIbis = periodes([ones(n,1)]); % Double or duration (period units)
    COIbis(:) = nan;
    COIbis(sum(COI)~=0) = periodes(a); % Only time steps of affected coefficients
    COIf = COIbis;
else
    COIf = periodes(a);
end

% Correcting initial values
if limit_C > 1 && limit_L ~= 0
    xlim = [limit_C+1 limit_C+2];
    ylim = [COIf(limit_C+1) COIf(limit_C+2)];
    pts = 1:limit_C;
    COIf(pts) = interp1(xlim,ylim,pts,'linear','extrap');
    pts = n-fliplr(pts)+1;
    xlim = [n-limit_C-1 n-limit_C];
    ylim = [COIf(n-limit_C-1) COIf(n-limit_C)];
    COIf(pts) = interp1(xlim,ylim,pts,'linear','extrap');
end
