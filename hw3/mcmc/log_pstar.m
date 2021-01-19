function logp = log_pstar(log_omega, mm, pie, mu1, mu2, log_sigma1, log_sigma2, xx, vv)
%LOG_PSTAR log joint probability up to a constant for the MLSS MCMC practical
%
%     logp = log_pstar(log_omega, mm, pie, mu1, mu2, log_sigma1, log_sigma2, xx, vv)
%
% Inputs:
%       log_omega 1x1 log resonant frequency
%              mm 1x1 black hole position
%
%   parameters of p(log(A)):
%             pie 1x1 mixing proportion for mixture of two Gaussians
%             mu1 1x1 mean 1
%             mu2 1x1 mean 2
%      log_sigma1 1x1 ...
%      log_sigma2 1x1 
%
%   data:
%              xx Nx1 object positions
%              vv Nx1 object velocities
%
% Outputs:
%           logp  1x1 log joint probability up to a constant

% Iain Murray, September 2009

log = @reallog; % Useful debugging aid, don't allow complex numbers to be created.

% Note: I've dropped some constant terms (e.g. p(phi)=1/(2*pi), sqrt(2*pi)).

N = length(xx);
sigma1 = exp(log_sigma1);
sigma2 = exp(log_sigma2);
omega = exp(log_omega);

% The "prior". Cheat slightly by getting some information about sane scaling
% from the data:
x_mu = mean(xx);
x_std = std(xx);
ext = max(xx) - min(xx);
log_ext = log(ext);
% Forbid extreme settings. Fairly vague prior. One could put more thought into this.
forbidden = ...
        (pie < 0) || (pie > 1) + ...
        (abs(mm - x_mu) > 10*x_std) +...
        (abs(mu1 - log_ext) > 20) + ...
        (abs(mu2 - log_ext) > 20) + ...
        (abs(log_sigma1) > log(20)) + ...
        (abs(log_sigma2) > log(20)) + ...
        (abs(log_omega) > 20);
if forbidden
    logp = -Inf;
    return;
end

log_A = 0.5*log((xx - mm).^2 + (vv/omega).^2);

Norm = @(x,m,s) exp(-0.5*(x-m).^2/(s*s)) / s;
%log_prior = sum(log(Norm(log_A,mu1,sigma1))); % Less flexible model with one component
log_prior = sum(log(pie*Norm(log_A,mu1,sigma1) + (1-pie)*Norm(log_A,mu2,sigma2)));
log_like = -2*sum(log_A) - N*log(omega);
logp = log_like + log_prior;

