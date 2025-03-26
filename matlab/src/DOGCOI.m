function coival = DOGCOI(method,m,s2,Fact_Fourier,varargin)
% Compute COI using integration

%%% Inputs
% - method (char): metric to define qcoi, 'integral' (area) or 'energy' (squared area)
% - m (double): order of derivation of the DOG function (>0)
% - s2 (double): Variance of the Haar or Gaussian function used to define wavelets
% - Fact_Fourier (double): Fourier factor, defined as the ratio of period to scale
% Optional argument:
% 1) qcoi (double): percentage of wavelet (squared) area outside signal edges

% Initialize
syms xvar
qcoi = 0.02;
if nargin==5
    qcoi = varargin{1}; 
end
if strcmpi(method(1:2),"L1") || strcmpi(method(1:3),'int')
    method = 'integral'; % Compute coi based on int(-Inf,Inf)|f(x)|dx
elseif strcmpi(method(1:2),"L2") || strcmpi(method(1:4),'ener')
    method = 'energy'; % Compute coi based on int(-Inf,Inf)|f(x)|^2dx
else 
    error('Use integral or energy for normalisation to adopt to compute COI')
end

% Function to integrate in symbolic notation
expr = diff(exp(-xvar^2/2/s2),m); 

% Evaluate and sort roots of DOG (given by those of Hermite polynomial)
if m<6
    sols = sort(eval(solve(hermiteH(m, xvar/sqrt(2*s2))==0,xvar)));
else
    sols = sort(eval(vpasolve(hermiteH(m, xvar/sqrt(2*s2))==0,xvar)));
end

% Computation of target = Value to achieve for integral of abs(psi) (squared or not) from -Inf to -b/a
if strcmp(method,'integral')
   if m<6 % Int(-Inf,Inf)f(x)dx analytically solved
        vals = [2 4/sqrt(exp(1)) 8/sqrt(exp(3))+2 4*sqrt(6)/exp(1.5+sqrt(1.5))*(sqrt(3+sqrt(6))+sqrt(3-sqrt(6))*exp(sqrt(6))) 2/exp(2.5+sqrt(2.5))*(16+8*sqrt(10)+(8*sqrt(10)-16)*exp(sqrt(10)))+6]; 
        cst = vals(m)/s2^((m-1)/2); % Int(-Inf,Inf)|f(x)|dx 
   else % If Int(-Inf,Inf)f(x)dx not known
        vals2 = matlabFunction(abs(diff(exp(-xvar^2/2/s2),m))); 
        cst = integral(vals2,-Inf,Inf,'RelTol',1e-15,'AbsTol',1e-15); % Int(-Inf,Inf)|f(x)|dx
   end
elseif strcmp(method,'energy')
    cst = gamma(m+0.5)/(s2^(m-0.5)); % Int(-Inf,Inf)|f(x)|^2dx
    expr = expr^2; % Adjusting function for energy (integration of squared expression) % expr = expr.^2;
end
target = qcoi*cst; % qcoi * integral from -Inf to +Inf of mother wavelet (squared or not)

% Computation of -b/a, upper bound of integral of abs(psi), for which qcoi percent of total integral of abs(psi) is achieved
k = 1; % Index of solutions to consider
value = 0; % Value to add to integral if seached value after first root
Limit_Inf = -Inf ;
Limit_Sup = sols(k) ;
func = matlabFunction(expr);
addv = abs(integral(func,Limit_Inf,Limit_Sup)); % Value to add to integral to check if integral between limits is > target
algo_init = [sols(k)-2*max(diff(sols)),Limit_Sup]; % Initial step for vpasolve

% Searching where -b/a is found : before first root (k=1, no loop), between first and second (k=2, one loop), ...
while value+addv < target
    k = k+1;
    value = value + addv;
    Limit_Inf = sols(k-1) ;
    Limit_Sup = sols(k) ;
    addv = abs(integral(func,Limit_Inf,Limit_Sup));
    algo_init = [Limit_Inf Limit_Sup]; % Define bounds for vpasolve
end

if isempty(sols(k)-max(diff(sols))) % Restrict bounds (for m=1)
    algo_init = [-10 0];
end

% Solving for -b/a (when (squared) area under wavelet = target)
coival = eval(vpasolve(abs(int(simplify(expr),Limit_Inf,xvar))==target-value,xvar,algo_init)); 
coival = -Fact_Fourier/coival; % -(-b/a) since t = b/a