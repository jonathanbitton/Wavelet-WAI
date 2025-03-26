function [wavsig] = significance(period,dt,ftype,signal,varargin)
% Computing significance threshold of wavelet coefficients for each scale
% assuming a red background noise.
% - Alpha level set to 0.05 (prob = 0.95) by default. To set another prob
% value, use significance(fper2,Ts,ftype,y,'prob',value)
% - lag1 autocorrelation computed automatically. If computation fails, 
% specify its value using significance(fper2,Ts,ftype,y,'lag1',value)

% Reference: Torrence, C. and G. P. Compo, 1998: A Practical Guide to
%            Wavelet Analysis. <I>Bull. Amer. Meteor. Soc.</I>, 79, 61-78.

if nargin>8
    error('Too many input arguments')
end

% Theroetical Fourier spectrum
norm_freq = dt./period;

% Degrees of freedom (real wavelets = 1)
if strcmpi(ftype(1:3),'MOR')
    dof = 2;
else
    dof = 1;
end

% Setting alpha level 
prob = 0.95;
if any(strcmpi(varargin,'prob'))
    ind = find(strcmpi(varargin,'prob'));
    prob = varargin{ind+1};
end
lag1=[];
if any(strcmpi(varargin,'lag1'))
    ind = find(strcmpi(varargin,'lag1'));
    lag1 = varargin{ind+1};
end

% Estimation of AR1 factor and signal variance
if isempty(lag1)
    lag1 = ar1nv(signal);
    if isnan(lag1)
        error("AR1 autocorrelation estimation %s. Manual estimation required (using arcov or arburg). \n Enter its value using significance(fper2,Ts,ftype,y,'lag1',value)",'failed')
    end
end
variance = var(signal);


% Computing significance
P = (1-lag1^2)./(1+lag1^2-2*lag1*cos(2*pi*norm_freq));
P = P*variance;
X2 = chi2inv(prob,dof);
wavsig = P*X2/dof;


function [g,a]=ar1nv(x)
% AR1NV - Estimate the parameters for an AR(1) model
% Syntax: [g,a]=ar1nv(x);
%
% Input: x - a time series.
%
% Output: g - estimate of the lag-one autocorrelation.
%         a - estimate of the noise variance.

% (c) Eric Breitenberger

x=x(:);
N=length(x);
m=mean(x);
x=x-m;

% Lag zero and one covariance estimates:
c0=x'*x/N;
c1=x(1:N-1)'*x(2:N)/(N-1);

g=c1/c0;
a=sqrt((1-g^2)*c0);