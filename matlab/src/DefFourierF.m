function [Fact_Fourier,ftype,s2,w0] = DefFourierF(ftype,varargin)
% Formatting ftype and deriving Fourier Factor, defined as the ratio of period to scale
%%% Inputs:
% - ftype (char): Analyzing function used among 'Morlet', 'Mexhat', 'DOGx', 'Gauss', 'Haar'
% Optional arguments using the format (...,'Argument',Value,'Argument',Value,...):
% 'norm' (char): Normalization convention, among 'L1' and 'L2'
% 's2' (double): Variance of the Haar or Gaussian function used to define wavelets
% 'w0' (double): Central angular frequency of the Morlet wavelet (ignored if ftype not set to Morlet)

[ftype,norm,s2,w0] = GetArgs({'ftype','norm','s2','w0'},'ftype',ftype,varargin{:});

% Defining Fourier Factor
switch ftype(1:3)
    case 'MOR' % Morlet wavelet
        if strcmpi(norm,'l1') % L1 norm
            Fact_Fourier = 2*pi/w0;
        else % L2 norm
            Fact_Fourier = 4*pi*sqrt(s2)/(sqrt(s2)*w0+sqrt(2+s2*w0^2)); 
        end
    case 'GAU' % Gauss function
        if strcmpi(norm,'l1') % L1 norm
            warning("Fourier Factor undefined for norm 'L1' using the Gaussian function. Returning norm 'L2' value")
        end
        Fact_Fourier = 2*pi*sqrt(s2*2);
    case 'DOG' % Gaussian derivatives 
        m = sscanf(ftype,'DOG%d'); % Order of the derivative
        if strcmpi(norm,'l1')
            Fact_Fourier = 2*pi*sqrt(s2)/sqrt(m);
        else
            Fact_Fourier = 2*pi*sqrt(s2)/sqrt(m+1/2);
        end
    case 'HAA' % Haar wavelet
        if strcmpi(norm,'l1')
            Fact_Fourier = 2*pi/4.6622447;
        else
            Fact_Fourier = 2*pi/5.5729963;
        end
end
