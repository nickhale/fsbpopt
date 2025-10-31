function [P, Q, D, Err_lsqr] = pqd(x, F, w)
%PQD  Compute P, Q, and an FSBP discretisation.
% 
%   [P, Q, D] = PQD(X, F) computes the matrices P, Q, and D in an FSBP
%   operator for the nodes X and function space F on the domain [X(1),
%   X(end)]. The weights are determined by the optimisation procedure of
%   [1].
%
%   [P, Q, D] = PQD(X, F, W) is the same, but uses the specified weights W.
%
%   The code required chebfun as a dependency (www.chebfun.org).
%
% Nick Hale, Dec 2024, Stellenbosch University

% Note. The optimisation proceedure is modified a little. In particular,
% rather than solve a constrained least squares problem with enforced
% symmetry, the symmetry condition is instead enforced manually so that an
% unconstrained least squares problem can be solved instead.

F = chebfun(F, [x(1), x(end)]);
F = qr(F);
Fp = diff(F); 
Fx = F(x); Fpx = Fp(x); [n,m] = size(Fx);

if ( nargin < 3 ), w = []; end
if ( isempty(w) ) % 'Optimisation'
    Z = [Fx ; -Fpx];
    B = zeros(length(x)); B([1,end]) = [-1 1];
    R_t = (-0.5*B*Fx).';
    % Set up unconstrained least squares problem
    IDX = tril(ones(n), -1); IDX(logical(IDX)) = 1:n*(n-1)/2;
    IDX = [IDX - IDX' ; diag(n*(n-1)/2+(1:n))];
    A = zeros(m*n, n*(n+1)/2);
    for k = 1:n
        for j = 1:2*n
            if (~IDX(j,k)), continue, end
            A((k-1)*m+(1:m), abs(IDX(j,k))) = sign(IDX(j,k))*Z(j,:);
        end
    end
    sol = lsqminnorm(A,R_t(:));            % Least-squares solution
    % sol = A\R_t(:);            % Least-squares solution
    Err_lsqr = norm(A*sol-R_t(:));         % Sanity check
    w = sol(n*(n-1)/2+(1:n)); P = diag(w); % Extract w nd P.
    
else % 'Brute force'
    P = diag(w); % w is given
    % Set up unconstrained least squares problem
    IDX = tril(ones(n), -1); IDX(logical(IDX)) = 1:n*(n-1)/2; IDX = IDX - IDX';
    A = zeros(m*n, n*(n-1)/2);
    for k = 1:n
        for j = 1:n
            if (~IDX(j,k)), continue, end
            A((k-1)*m+(1:m), abs(IDX(j,k))) = sign(IDX(j,k))*Fx(j,:);
        end
    end
    B = zeros(length(x)); B([1,end]) = [-1 1];
    R = P*Fpx-0.5*B*Fx; R_t = R';   
    sol = lsqminnorm(A, R_t(:));         % Least-squares solution
    Err_lsqr = norm(A*sol-R_t(:));       % Sanity check
end

% Construct S, Q, and D.
S = tril(ones(n), -1); % Lower triangular mask
S(logical(S)) = sol(1:(n*(n-1)/2));
S = -S + S.';
Q = B/2 + S;
D = diag(1./w)*Q;

end