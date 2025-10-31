function [P, Q, D, Err_lsqr] = pqd_orig(x, F)

if (nargin == 0)
    % Test case
    x = linspace(0,1,4)';
    F = @(x) [x.^0, x, exp(x)]; % Basis
end

N = 4;
x0 = x(1); x1 = x(end);
F = chebfun(F, [x(1), x(end)]);
Fp = diff(F);

% Constructing Vandermonde and the corresponding derivatives
V = F(x);
Vp = Fp(x);

% Constructing Q and P
B = zeros(N); B(1,1) = -1; B(end,end) = 1; % boundary matrix
size(B)
size(V)
r1 = -0.5*B*V; r1 = r1(:);                 % RHS vec for the first equation

[C,K] = comm_matrix(N);                    % Communication matrices

A = [kron(V',speye(N)), kron(-Vp',speye(N))*K];
E = [C+speye(N^2),      zeros(N^2,N);
     zeros(1,N^2),      ones(1,N)];        % Ensures that constants are integrated exactly                                         
r2 = [zeros(N^2,1); x1-x0 ];

% E = [C+speye(N^2),      zeros(N^2,N)]; 
% r2 = zeros(N^2,1);

% linear problem:  Ay = r1
% constraits:      Ey = r2 and P>0
opts = optimoptions('lsqlin', 'Algorithm', 'interior-point', 'TolFun' ,1e-14, 'TolX' ,1e-14);
y = lsqlin(A,r1,[],[],E,r2,[],[],[],opts); % Solving the least square problem
Err_lsqr = norm(A*y-r1, inf);

% Extracting and computing S, Q, P, and D:
S = reshape(y(1:N^2),N,N);
Q = S + 0.5*B;            
P  = diag(y(N^2+1:end));
D = diag(1./y(N^2+1:end))*Q;

end

function [C,K] = comm_matrix(N)
% Commutation matrix for S
I = reshape(1:N*N, [N, N]);   % initialize a matrix of indices
I = I';                       % transpose it
I = I(:);                     % vectorize the required indices
C = speye(N*N);               % Initialize an identity matrix
C = C(I,:);                   % Re-arrange the rows of the identity matrix

%communication matrix for P
I = reshape(1:N*N, [N, N]);
K = speye(N*N);               % Initialize an identity matrix
K = K(diag(I),:)';            % Re-arrange the rows of the identity matrix
end