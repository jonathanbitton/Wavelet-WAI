function [s2rec,cpsi] = Compute_stat(wave1,wave2,scales,dj,dt,ftype,norm,varargin)
% Computing reconstructed (co)variance for a wavelet transform

%%% Inputs
% - wave1 (double or complex): wavelet coefficients of the first signal
% - wave2 (double or complex): wavelet coefficients of the second signal (for variance computation, input the same coefficients as wave1)
% - scales (double): scales used (before conversion to periods or frequencies)
% - dt (double): Adimensionnal sampling period
% - dj (double): Frequency resolution
% - ftype (char): Analyzing function used among 'Morlet', 'Mexhat', 'DOGx', 'Gauss', 'Haar'
% - norm (datetime or double): normalization adopted for the wavelet transform
% Optional arguments using the format (...,'Argument',Value,'Argument',Value):
% 1) 's2' (double): Variance of the Gaussian function used to define wavelets
% 2) 'w0' (double): Central angular frequency of the Morlet wavelet (ignored if ftype not set to Morlet)
% Default settings are s2=1 and w0=6

%%% Outputs
% s2rec: reconstructed (co)variance
% cpsi: wavelet admissibility constant

% Importing data
% Extract s2
s2 = 1;
if any(strcmpi(varargin,'s2'))
    ind = find(strcmpi(varargin,'s2'));
    s2 = (varargin{ind+1});
    if ~isnumeric(s2)
        error("s2 must be a double, not a %s",class(s2))
    end
end

% Extract w0
w0 = 6;
if any(strcmpi(varargin,'w0'))
    ind = find(strcmpi(varargin,'w0'));
    w0 = varargin{ind+1};
    if ~isnumeric(w0)
        error("w0 must be a double, not a %s",class(w0))
    end
end

% Convert ftype and verify formats
[ftype,dt,norm] = GetArgs({'ftype','dt','norm'},'ftype',ftype,'ts',dt,'norm',norm);

% Define cpsi and signal length
[cpsi,fac] = define_wavcst(ftype,norm,s2,w0);
L = size(wave1,2);

% Compute cross products and convert to real value if necessary (only for covariance)
cross = wave1.*conj(wave2);
if ~isreal(cross) % for covariance (real signals only)
    cross = real(cross);
end

% Reconstrcut (co)variance
if contains(norm,'2')
    % Reconstruction (CWT theory)
    s2rec = fac*dj*dt*log(2)/(cpsi)/L;
    s2rec = s2rec*sum(sum(cross./scales'));
elseif contains(norm,'1')
    % Reconstruction (CWT theory)
    s2rec = fac*dj*log(2)/(cpsi)/L;
    s2rec = s2rec*sum(sum(cross));
end

end

function [cpsi,fac] = define_wavcst(ftype,norm,s2,w0)
    switch ftype(1:3)
        case 'MOR' % Ondelette de Morlet
            if strcmp(norm,'L1') % constants divided by (2*pi*s2)
                cst=2;
            else
                cst=(4*s2*pi)^(1/4);
            end
            wav = @(w)abs(cst^2*exp(-s2*(w-w0).^2))./w; % |ψ(t)|²/w
            fac = 2;
        case 'GAU' % Gauss function
            error('Reconstruction not implemented for gaussian function as it is not a wavelet')
        case 'DOG' % Gaussian derivatives 
            m = sscanf(ftype,'DOG%d'); % Order of the derivative
            NormL1 = 2; 
            if strcmp(norm,'L1')
                if m<6
                    vals = [2 4/sqrt(exp(1)) 8/sqrt(exp(3))+2 4*sqrt(6)/exp(1.5+sqrt(1.5))*(sqrt(3+sqrt(6))+sqrt(3-sqrt(6))*exp(sqrt(6))) 2/exp(2.5+sqrt(2.5))*(16+8*sqrt(10)+(8*sqrt(10)-16)*exp(sqrt(10)))+6]; 
                    cst = vals(m)/s2^((m-1)/2);
                else
                    syms xvar
                    vals2 = matlabFunction(abs(diff(exp(-xvar^2/2/s2),m)));
                    cst = integral(vals2,-Inf,Inf,'RelTol',1e-15,'AbsTol',1e-15);
                end
                cst = -(1i)^m*NormL1/cst*sqrt(2*pi*s2);
            else 
                cst = -(1i)^m*sqrt(s2^(m-0.5)/gamma(m+0.5))*sqrt(2*pi*s2);
            end
            wav = @(w)abs(cst^2*exp(-s2*(w.^2)).*(w.^(2*m-1)));
            fac = 1;
        case 'HAA' % Haar wavelet
            NormL1 = 2;
            if strcmp(norm,'L1')
                cst = NormL1;
            else
                cst = 1;
            end
            wav = @(w) cst^2*(16 * (sin(w/4)).^4) ./ (w .^ 3);
            fac = 1;
    end
    cpsi = integral(wav,0,Inf);
end