function [x, w] = gglq(G, dom, opts, tol)
%QQLQ Generalised Gauss Lobatto quadrature nodes and weights.
%
%   [X, W] = GGLQ(G, DOM) computes the generalsed Gauss-Lobatto quadrature
%   nodes and weights for the function space G on the domain DOM.
%
%   [X, W] = GGLQ(G, DOM, OPTS) allows passing different optiond to the
%   solver. Currently the only valid OPTS is the string 'ITER' which
%   displays information about convergence of the algorithm.
%
%   [X, W] = GGLQ(G, DOM, OPTS, TOL) allows specifying a desired tolerance
%   for the computation. The defaul is 1e-14.
%
%
%   This implemention uses a Lobatto variant of the algorithm of Ma et al,
%   [SINUM 1996] with elements of the modifications by Yarvin and Rokhlin
%   [SISC 1998]. It requires Chebfun (www.chebfun.org).
%
% Nick Hale, 22 July 2025. Stellenbosch University

% Warning / error codes:
% fail = 1 -> NaN/Inf encounterd
% fail = 2 -> x outside [a,b] or some w_k < 0
% fail = 3 -> max iterations exceeded

if ( nargin < 4 ), tol = 1e-14; end
if ( nargin < 3 ), opts = []; end

G = chebfun(G, dom);          % Make a Chebfun for convenience
G = qr(G); N = size(G,2)/2;   % Orthogonalize
S = sum(G).'; Gp = diff(G);   % Integrate and differentials

n = 2; fail = false;
while ( n <= N )
    % Extract first 2n terms.
    Gn = G(:,1:2*n); Sn = S(1:2*n); Gnp = Gp(:,1:1:2*n);
    % Initial guess for nodes:
    if ( n == 2 )
        c = [dom(1) ; mean(dom) ; dom(2)]; 
    else % Use interlacing property.
        c = [dom(1) ; (c(2:end) + c(1:end-1))/2 ; dom(2)];
    end
    S0 = sum(Gn(c)).';  

    dt = .1; t = 0; succ = 0; kmax = 20; % initialise 'time stepping'
    while ( dt > eps )                   % time stepping
        fail = false;
        x = c; 
        St = t*Sn + (1-t)*S0;
        
        for k = 1:20 % Newton iteration
            % Evaluate the basis functions at the current estimate
            A = [Gn(x) ; Gnp(x(2:n))];
            % Compute the Lagrange-Hermite coefficients
            warning off, sol = A'\St; w = sol(1:n+1); warning on
            dx = sol(n+2:2*n)./w(2:n); % Newton update
            x(2:n) = x(2:n) + dx;
            I = Gn(x)'*w - St;
            if ( strcmpi(opts, 'iter') )
                % Optionally display convergence
                % fprintf('n\t t\t\t\t\t dt \t\t k \t\t dx \t\t\t I\n')
                fprintf('%d\t%16.16f\t%16.16f\t%d\t%16.16f\t%16.16f\n', ...
                    n, t, dt, k, norm(dx, inf), norm(I, inf))
                % disp([n t dt k norm(dx, inf) norm(I, inf)])
            end
            if ( any(isnan(dx)) )
                fail = 1; break % break on failure
            elseif ( any(x < dom(1)) || any(x > dom(2))  || any(w < 0)) 
                fail = 2; break % break on failure              
            end
            if ( norm(dx, inf) < k*tol || norm(I, inf) < tol)
                break           % break on success
            end 
        end
        if ( ~fail && k == kmax ), fail = 3; end

        % Adaptivity
        if ( ~fail )
            succ = succ+1;      % increment success counter
            c = x;              % accept new solution
            if ( succ == 4 )
                dt = 1.5*dt;    % increase 'time step'
                succ = 0;       % reset counter
            end
            if ( t == 1 ), break ,end
            t = min(t + dt,1);  % increment time 
        else
            succ = 0;           % reset succes counter
            dt = dt/2;          % decrease 'time step'
            t = max(t-dt,0);    % go back in time.
        end

    end

    if ( dt < eps || k == kmax )
        warning(['Failed for n = ', int2str(n), ...
            ' of ' int2str(N) ' at t = ' num2str(t) '.' ...
            ' Failure code:' int2str(fail)])
    end
    n = n + 1;
    
end

if ( dt < eps ), warning('We probably failed.'); end
if ( norm(w'*Gn(x) - sum(Gn), inf) > 1e-10 ), warning('We failed.'); end

end