%% Example 1a
clc

dom = [0 1];
F = @(x) [1+0*x, x, exp(x)];
G = @(x) [1+0*x, x, exp(x), x.*exp(x), exp(2*x), x.^2];

[x, w, P, D, err] = gsbpopt(F, dom, G);
disp(table(x, w))
D
num2str(D, '%1.10f')

% Error checks
err_type = ['quad' ; 'lsqr' ; 'fsbp' ;'aprx'];
disp(table(err_type, err))


%% Example 1b
clc

dom = [0 1];
F = @(x) [1+0*x, x, exp(x)];
G = @(x) [1+0*x, x, exp(x), x.*exp(x), exp(2*x)]; 

x = linspace(0, 1, 5).';
% [P, Q, D, Err_lsqr] = pqd(x, F);
[P, Q, D, Err_lsqr] = pqd_orig(x, F);
w = diag(P);
disp(table(x, w))
D

% Error checks
G = chebfun(G, dom);
err = [];
err(1) = norm(w'*G(x) - sum(G), inf);   % Err_quad
err(2,1) = Err_lsqr;                    % Err_lsqr
F = chebfun(F, dom); Fp = diff(F);
err(3,1) = norm(D*F(x)-Fp(x), inf);       % Err_FSBP
f = @(x) exp(-x.^2); fp = @(x) -2*x.*exp(-x.^2);
err(4)= norm(D*f(x)-fp(x), inf);        % Err_aprx
err_type = ['quad' ; 'lsqr' ; 'fsbp' ;'aprx'];
disp(table(err_type, err))

%% Example 1c
clc

dom = [0 1];
F = @(x) [1+0*x, x, exp(x)];
G = @(x) [1+0*x, x, exp(x), x.*exp(x), exp(2*x)]; 

x = linspace(0, 1, 4).';
% [P, Q, D, Err_lsqr] = pqd(x, F);
[P, Q, D, Err_lsqr] = pqd_orig(x, F);
w = diag(P);
disp(table(x, w))


% Error checks
G = chebfun(G, dom);
err = [];
err(1) = norm(w'*G(x) - sum(G), inf);   % Err_quad
err(2,1) = Err_lsqr;                    % Err_lsqr
F = chebfun(F, dom); Fp = diff(F);
err(3) = norm(D*F(x)-Fp(x), inf);       % Err_FSBP
f = @(x) exp(-x.^2); fp = @(x) -2*x.*exp(-x.^2);
err(4)= norm(D*f(x)-fp(x), inf);        % Err_aprx
err_type = ['quad' ; 'lsqr' ; 'fsbp' ;'aprx'];
disp(table(err_type, err))

%% Example 1d
clc

dom = [0 1];
F = @(x) [1+0*x, x, exp(x)];
G = @(x) [1+0*x, x, exp(x), x.*exp(x), exp(2*x)]; 

x = lobpts(4); x = (x+1)/2*diff(dom) + dom(1);
[P, Q, D, Err_lsqr] = pqd(x, F);
w = diag(P);
disp(table(x, w))
D

% Error checks
G = chebfun(G, dom);
err = [];
err(1) = norm(w'*G(x) - sum(G), inf);   % Err_quad
err(2,1) = Err_lsqr;                    % Err_lsqr
F = chebfun(F, dom); Fp = diff(F);
err(3) = norm(D*F(x)-Fp(x), inf);       % Err_FSBP
f = @(x) exp(-x.^2); fp = @(x) -2*x.*exp(-x.^2);
err(4)= norm(D*f(x)-fp(x), inf);        % Err_aprx
err_type = ['quad' ; 'lsqr' ; 'fsbp' ;'aprx'];
disp(table(err_type, err))

%% Example 2: Bessel functions
clc

tol = 1e-10;
dom = [0 15];
xc = chebfun('x', dom);
F = besselj(0:15,xc);
F = @(x) feval(F, x);

[x, w, P, D, err] = gsbpopt(F, dom, [], tol);
disp(table(x, w))
length(x)

% Error checks
err_type = ['quad' ; 'lsqr' ; 'fsbp' ;'aprx'];
disp(table(err_type, err))


%% Example 2b: Bessel functions (equispaced)
clc

tol = 1e-10;
dom = [0 15];
xc = chebfun('x', dom);
F = besselj(0:15,xc);
F = @(x) feval(F, x);

F_ = chebfun(F, dom); Fp_ = diff(F_);
% err = norm(D*F_(x)-Fp_(x), inf);       % Err_FSBP

% for N = 10:200
for N = 169
    x = linspace(dom(1), dom(2), N).';
    [P, Q, D] = pqd(x, F);
    w = diag(P);
    if ( all(w > 0) )
        err = norm(D*F_(x)-Fp_(x), inf)       % Err_FSBP
        if ( err < 1e-8 )
            break, 
        end
    end
end
disp(table(x, w))
length(x)

% Error checks
errq = []; GG = {};
for k = 1:size(F_,2)
    for j = 1:size(F_,2)
        G = diff(F_(:,j)*F_(:,k));
        errq(j,k) = abs(w'*G(x) - sum(G));
        GG{j,k} = G;
    end
end
err = [];
err(1) = norm(errq(:), inf);   % Err_quad
err(2,1) = Err_lsqr;                    % Err_lsqr
F = chebfun(F, dom); Fp = diff(F);
err(3) = norm(D*F(x)-Fp(x), inf);       % Err_FSBP
f = @(x) exp(-x.^2); fp = @(x) -2*x.*exp(-x.^2);
err(4)= norm(D*f(x)-fp(x), inf);        % Err_aprx
err_type = ['quad' ; 'lsqr' ; 'fsbp' ;'aprx'];
disp(table(err_type, err))

%% Example 2b: Bessel functions (Gauss-Lobatto)
clc

tol = 1e-10;
dom = [0 15];
xc = chebfun('x', dom);
F = besselj(0:15,xc);
F = @(x) feval(F, x);

F_ = chebfun(F, dom); Fp_ = diff(F_);
% err = norm(D*F_(x)-Fp_(x), inf);       % Err_FSBP

for N = 10:200
    x = lobpts(N); x = (x+1)/2*diff(dom) + dom(1);
    % [P, Q, D] = pqd(x, F);
    [P, Q, D] = pqd_prince(x, F);
    w = diag(P);
    if ( all(w > 0) )
        err = norm(D*F_(x)-Fp_(x), inf);       % Err_FSBP
        if ( err < tol)
            break, 
        end
    end
end
disp(table(x, w))
length(x)

% Error checks
errq = []; GG = {};
for k = 1:size(F_,2)
    for j = 1:size(F_,2)
        G = diff(F_(:,j)*F_(:,k));
        errq(j,k) = abs(w'*G(x) - sum(G));
        GG{j,k} = G;
    end
end
err = [];
err(1) = norm(errq(:), inf);   % Err_quad
err(2,1) = Err_lsqr;                    % Err_lsqr
F = chebfun(F, dom); Fp = diff(F);
err(3) = norm(D*F(x)-Fp(x), inf);       % Err_FSBP
f = @(x) exp(-x.^2); fp = @(x) -2*x.*exp(-x.^2);
err(4)= norm(D*f(x)-fp(x), inf);        % Err_aprx
err_type = ['quad' ; 'lsqr' ; 'fsbp' ;'aprx'];
disp(table(err_type, err))


