function [samples, arate] = dumb_metropolis(init, log_ptilde, iters, sigma, varargin)
%DUMB_METROPOLIS explore an unnormalized distribution. Eg code for tutorial
%
%     samples = dumb_metropolis(init, log_ptilde, iters, sigma)
%
% Inputs:
%            init Dx1 An initial condition (an array of any shape with D elements) 
%      log_ptilde @fn function that takes inputs like init and returns
%                     the log-probability of the target density up to a constant.
%                     It can also be passed extra arguments specified by varargin
%           iters 1x1 Number of samples to gather
%           sigma 1x1 Step size for spherical proposals
%            ... + any extra args that will be passed to log_ptilde after the state
%
% Outputs:
%        samples  Dxiters states visited by the Markov chain
%          arate  1x1 acceptance rate. Tune sigma to get roughly a half (a bit less).

% Iain Murray, September 2009

D = numel(init);
samples = zeros(D, iters);

arate = 0;
state = init;
Lp_state = log_ptilde(state, varargin{:});
for ss = 1:iters
    % Propose
    prop = state + sigma*randn(size(state));
    Lp_prop = log_ptilde(prop, varargin{:});
    if log(rand) < (Lp_prop - Lp_state)
        % Accept
        arate = arate + 1;
        state = prop;
        Lp_state = Lp_prop;
    end
    samples(:, ss) = state(:);
end
arate = arate/iters;
