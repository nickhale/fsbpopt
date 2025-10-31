function [x, w, P, D, err] = fsbpopt(F, dom, G, tol)
%FSBPOPT Optimal nodes for generalised function space SBP operators
%
% [X, W] = FSBPOPT(F, DOM) computes the optimal nodes and weights for a SBP
% discretisation for the function space F on the domain DOM. 
%
% [X, W] = FSBPOPT(F, DOM, G) specifies the space G = (FF)'. If not given,
% this is calculated numerically.
%
% [X, W] = FSBPOPT(F, DOM, G, TOL) allows a user specified tolerance. The
% default is 1e-14.
%
% [X, W, P, D] = ... returns also the corresponding matrices P and D for
% the SBP operator.
%
% This code requires Chebfun as a dependency (www.chebfun.org).
%
% Nick Hale, July 2025, Stellenbosch University

% Parse inputs:
if ( nargin < 4 )
    tol = 1e-14;
end
if ( nargin < 3 )
    G = []; 
end
if ( nargin < 2 )
    % Default example
    dom = [0, 1];
    F = @(x) [1+0*x, x, exp(x)];
    G = @(x) [1+0*x, x, exp(x), x.*exp(x), exp(2*x), x.^2];
end
x = []; w = [];

% G is not given, so compute it numerically:
if ( isempty(G) )
    [G, s] = findG(F, dom, tol, {'rescale'});
    % singularvals = s % Uncomment to show what singular values are discarded
else
    G = chebfun(G, dom);
end

% Obtain x using generalisd Gauss quadrature
if ( isempty(x) )
    % Change 'none' to 'iter' to display convergence
    [x, w] = gglq(G, dom, 'none');
end
num_pts = length(x);

% Construct P, Q, and D
[P, Q, D, Err_lsqr] = pqd(x, F, w);
if ( isempty(w) ), w = diag(P);  end

% Error checks
if ( nargout == 0 || nargout > 4)
    err = [];
    err(1) = norm(w'*G(x) - sum(G), inf);   % Err_quad
    err(2,1) = Err_lsqr;                    % Err_lsqr
    F = chebfun(F, dom); Fp = diff(F);
    err(3) = norm(D*F(x)-Fp(x), inf);       % Err_FSBP
    f = @(x) exp(-x.^2); fp = @(x) -2*x.*exp(-x.^2);
    err(4)= norm(D*f(x)-fp(x), inf);        % Err_aprx
end

if ( nargout == 0 )
    disp(table(x, w))
    err_type = ['quad' ; 'lsqr' ; 'fsbp' ;'aprx'];
    disp(table(err_type, err))
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [G,s] = findG(F, dom, tol, opts) 
% Given a function space F, determine G = (FF)' numerically.
% Nick Hale, Jan 2025, Stellenbosch University

% This code attempts to find an orthogonal basis for the space G = (FF'). 
% It does so by 
%   i) forming all combinations of (F_j*Fk)', 
%  ii) computing a singular value decomposition of the result, 
% iii) keeping only left singular vecs with corresp. singular vals > tol
%  iv) if reqd, adding back largest truncated term to enforce an even no.

if ( isscalar(dom) ) , tol = dom;
elseif ( nargin < 3 ), tol = 1e-14; end
if ( nargin < 4 ), opts = []; end
F = chebfun(F, dom); [F, ~] = qr(F); % Orthogonalize F for stability
G = cell(size(F,2)*(size(F,2)+1)/2,1); l = 1;
for j = 1:size(F,2)                  % Construct G = (FF)'
    for k = j:size(F,2)
        G{l} = diff(F(:,j).*F(:,k)); l = l + 1;
    end
end
if ( any(strcmpi(opts,'force1')) ), G{end+1} = chebfun(1, dom); end
% Construct an orthogonal basis
G = [G{:}]; [G, S, ~] = svd(G); s = diag(S);
idx = abs(s./s(1))<tol;      % Discard small singular values...
if ( mod(sum(~idx), 2) )     % ... but enforce an even number of terms
    idx(find(idx, 1, 'first')) = false;
end
if ( any(strcmpi(opts,'rescale')) )
    G = G(:,~idx)*diag(s(~idx)); % Rescale? (This is questionable.)
end
s(idx) = -s(idx); s(s==0) = -realmin;    % For display purposes
end
