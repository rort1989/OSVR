function [model,history,z] = admm(A, e, lambda, mu, varargin)
% admm: Alternative Direction Method of Multipliers solver
% Author: Rui Zhao 2015.11.13.
% Reference: Boyd 2011, Distributed optimization and statistical learning via the alternating direction method of multipliers
%
% model = admm(Problem, mu) solve the following optimization problem using
% ADMM updates
%
%      minimize_{x,z}  0.5*x'*L*x + u'*((z)_+).^p
%      s.t. z = Ax + e
%
% Input:
%       A: data matrix
%       e: label vector
%       lambda: a vector or a scalar specifying the diagonal entries of L
%       mu: a vector storing weights
% 
% Output: 
%       model.w is the model parameter without bias term
%       model.b is the model parameter bias term
% 
% [model,history,z] = admm(Problem, mu) also returns 
%       history, a struct storing the ADMM update history and
%       z, auxilary variables introduced by ADMM
% 
% [model,history,z] = admm(Problem, mu, varargin) support optional input arguments:
%       'rho': value of augmented Lagrangian multipliers
%       'max_iter': maximum number iterations allowed before convergence
%       'bias': 1 if add bias term to features and 0 otherwise
%       'option': 1: use hingle loss i.e. p = 1, 2: use squared hinge loss i.e. p = 2
%       'ABSTOL': absolute tolerence value used to decide convergence
%       'RELTOL': relative tolerence value used to decide convergence
%       'early_stop': 1: stop update of x if meet convergence criteria; 0 otherwise
%
% Example: 
%       load('angry.mat');
%       [model,history,z] = admm(A, e, lambda, mu, 'rho', rho, 'max_iter', max_iter, 'option', option);
% 
% Example:
% admm() can be used to solve C-SVM with following formulation
% the primal problem of L2 regularized hinge or squared hinge loss linear C-SVM using ADMM
% SVM formulation:
%       \min_{w,b}  0.5*||w||^2 + \sum_{n=1}^N max(0,1-y_n(w^T x_n + b))^p
% Reference: 

% specify default values
rho = 1;
max_iter = 100;
bias = 0;
option = 1;
ABSTOL = 1e-3;
RELTOL = 1e-3;
early_stop = 0;
for argidx = 1:2:nargin-4
    switch varargin{argidx}
        case 'rho'
            rho = varargin{argidx+1};
        case 'max_iter'
            max_iter = varargin{argidx+1};
        case 'bias'
            bias = varargin{argidx+1};
        case 'option'
            option = varargin{argidx+1};   
        case 'ABSTOL'
            ABSTOL = varargin{argidx+1};   
        case 'RELTOL'
            RELTOL = varargin{argidx+1};   
        case 'early_stop'
            early_stop = varargin{argidx+1};   
    end
end

% initialize data
[N,D] = size(A); % N instances * D dimensional features
assert(length(e)==N,'Number of labels does not match data dimension');
A = [A ones(N,bias)];
assert(numel(lambda)==length(lambda),'lambda must be a scalar or a vector');
lambda = [lambda(:); ones(bias,1)];
lambda = diag(lambda);

% initialize variables
x = zeros(D+bias,1);  % first d entries are w, last entry is b
z = zeros(N,1);      % alternating variables
y = zeros(N,1);      % multipliers
% mu = gamma*mu; % change the values if you want to assign different weights to different samples
loss = A*x+e;

% different loss functions
loss(loss<0) = 0;
if option == 1 % L2-norm hinge loss    
    f = 0.5*(x')*lambda*x + mu'*loss;
elseif option == 2 % L2-norm squared hinge loss    
    f = 0.5*(x')*lambda*x + mu'*(loss.^2);
end

% main iterations
r_norm = zeros(1,max_iter);
s_norm = zeros(1,max_iter);
eps_pri = zeros(1,max_iter);
eps_dual = zeros(1,max_iter);
obj_res = zeros(1,max_iter);
obj = zeros(1,max_iter);
x_change = zeros(1,max_iter); x_flag = 0;
y_change = zeros(1,max_iter);
z_change = zeros(1,max_iter);

for it = 1:max_iter
    % update x
    if ~x_flag % in case x converged but not the objective function
        x_old = x;
        % l2 norm regularization
            if it == 1
                H = 1/rho*lambda*eye(D+bias) + A'*A; % (D+1)*(D+1)
                [U,S,V] = svd(H,0);
                Sinv = diag(1./diag(S));
                Hinv = V*Sinv*(U')*(A');
            end
            q = (z - y/rho - e);  % N*1
            x = Hinv*q;
    end
    x_change(it) = norm(x-x_old);
    % optional early stop
    if x_change(it) < ABSTOL && early_stop
        x_flag = 1;
    end
    
    % update z    
    z_old = z;
    Ax = A*x;
    loss = Ax + e;
    if option == 1 % hinge loss
        theta =  y/rho + loss - 0.5/rho*mu;  % N*1 vector
        a = mu/2/rho;
        z(theta>=0) = max(theta(theta>=0)-a(theta>=0),0);
        z(theta<0) = min(theta(theta<0)+a(theta<0),0);        
    elseif option == 2 % squared hinge loss
        theta =  y/rho + loss; % N*1 vector
        z(theta>=0) = rho.*theta(theta>=0)./(rho+2*mu(theta>=0));
        z(theta<0) = theta(theta<0);
    end
    z_change(it) = norm(z-z_old);
    
    % update y
    dy = rho*(loss - z);
    y = y + dy; % N*1 vector
    y_change(it) = norm(dy);    
    
    % check convergence 
    % objective convergence: check the change of objective function
    loss(loss<0) = 0;
    if option == 1        
        f_new = 0.5*(x')*lambda*x + mu'*loss;
    elseif option == 2
        f_new = 0.5*(x')*lambda*x + mu'*(loss.^2);
    end
    obj(it) = f_new;
    obj_res(it) = f_new - f; 
    f = f_new;
    % use residual convergence to decide if exit early
    r_norm(it) = norm(dy/rho);
    s_norm(it)  = norm(rho*(z - z_old));    % A'*
    eps_pri(it) = sqrt(N)*ABSTOL + RELTOL*max([norm(Ax) norm(-z) norm(e)]);
    eps_dual(it) = sqrt(D+bias)*ABSTOL + RELTOL*norm(y);  % A'* 
    if (r_norm(it) < eps_pri(it) && s_norm(it) < eps_dual(it))
         break;
    end
    if abs(r_norm(it)) < ABSTOL && abs(obj_res(it)) < ABSTOL
        break;
    end
end

% save results
model.w = x(1:D);
model.b = bias*x(D+bias);
history.iter = it;
history.r_norm = r_norm(1:it);
history.s_norm = s_norm(1:it);
history.eps_pri = eps_pri(1:it);
history.eps_dual = eps_dual(1:it);
history.obj_res = obj_res(1:it);
history.obj = obj(1:it);
history.rho = rho;
history.x_change = x_change(1:it);
history.y_change = y_change(1:it);
history.z_change = z_change(1:it);

end
