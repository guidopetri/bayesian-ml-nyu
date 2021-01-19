% This is my messy driver code which loads in the data and actually runs the
% sampler on the model. As I was playing around interactively, at this stage
% it's just a script that I've been tweaking and re-running.

% Load data
ws = load('astro_data');
xx = ws.xx;
vv = ws.vv;
clear('ws');

% Initial condition
mu = mean(xx); 
ext = max(xx) - min(xx);
log_omega = log(mean(abs(vv)) / ext); % Something with the right units
mm = mu;
pie = 0.5;
mu1 = log(0.5*ext*rand);
mu2 = log(0.5*ext*rand); 
log_sigma1 = 1;
log_sigma2 = 1;
init = [log_omega; mm; pie; mu1; mu2; log_sigma1; log_sigma2];

% Was careful initialization necessary? For slice sampling? For Metropolis?
% Here's an alternative:
%init = rand(size(init));

% Run sampler
wrapper = @(args) log_pstar(args{:}, xx, vv);
logdist = @(x) wrapper(num2cell(x));
assert(~isinf(logdist(init))); % Need a feasible point to begin with
%if true % Run slice sampling
if false % Run Metropolis
    % Slice sample
    % Often it's best to set widths on the large side and set step_out to false.
    % But I was in too much of a hurry to think about setting widths carefully
    % and step_out=true is more robust.
    S = 1e3;
    step_out = true;
    widths = 1;
    samples = slice_sample(S, 0, logdist, init, widths, step_out);
    %save('slice_samples', 'samples');
    burn = 100;
else
    S = 1e5;
    [samples, arate] = dumb_metropolis(init, logdist, S, 0.017);
    %save('metropolis_samples', 'samples');
    burn = 10000;
    arate
end

% Report some results
omega_s = exp(samples(1,burn:end));
fprintf('omega = %s\n', ...
        errorbar_str(mean(omega_s), std(omega_s)));
fprintf('m = %s\n', ...
        errorbar_str(mean(samples(2,:)), std(samples(2,:))));
clf;
subplot(2,2,1); hist(omega_s, 30);
subplot(2,2,2); hist(samples(2,burn:end), 30);
subplot(2,2,3); plot(samples(1,:), samples(2,:));
subplot(2,2,4); plot(samples(1,:));

