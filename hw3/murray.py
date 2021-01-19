import numpy as np


def log_pstar(log_omega, mm, pie, mu1, mu2, log_sigma1, log_sigma2, xx, vv):
    """
    Code adapted from Iain Murray's matlab code.
    
    in: 
      scalars:
        log_omega (log resonant frequency)
        mm (black hole position)
        pie (gaussian mixture proportion)
        mu1 (mean 1)
        mu2 (mean 2)
        log_sigma1
        log_sigma2
      arrays:  
        xx (object positions, (N, 1) array)
        vv (object velocities, (N, 1) array)
    out:
      scalar `logp`, the log joint probability "up to a constant"
    """
    N = xx.shape[0]
    sigma1 = np.exp(log_sigma1)
    sigma2 = np.exp(log_sigma2)
    omega = np.exp(log_omega)
    
    x_mu = xx.mean()
    x_std = xx.std()
    ext = xx.max() - xx.min()
    log_ext = np.log(ext)
    
    forbidden_conditions = [pie < 0,
                            pie > 1,
                            np.abs(mm - x_mu) > 10 * x_std,
                            np.abs(mu1 - log_ext) > 20,
                            np.abs(mu2 - log_ext) > 20,
                            np.abs(log_sigma1) > np.log(20),
                            np.abs(log_sigma2) > np.log(20),
                            np.abs(log_omega) > 20,
                            ]
    
    if any(forbidden_conditions):
        return -np.inf
    
    log_A = 0.5 * np.log((xx - mm) ** 2 + (vv / omega) ** 2)
    
    def norm(x, m, s):
        return np.exp(-0.5 * ((x - m) ** 2) / (s ** 2)) / s
    
    # other, less flexible model:
    # log_prior = np.sum(np.log(norm(log_A, mu1, sigma1)))
    log_prior = np.sum(np.log(pie * norm(log_A, mu1, sigma1)
                              + (1 - pie) * norm(log_A, mu2, sigma2)))
    log_like = -2 * log_A.sum() - N * np.log(omega)
    logp = log_like + log_prior
    
    return logp


def slice_sample(N, burn, logdist, x, widths, step_out, *args):
    """
    Code adapted from Iain Murray's matlab code.
    
    in:
      scalar:
        N (number of samples)
        burn (burn-in period length)
        step_out (boolean; set to true if widths are sometimes too small)
        *args (other arguments passed to logdist)
      array:
        x (initial state, (D, 1) shape)
        widths (step sizes for slice sampling; (D, 1) or scalar shape)
      function:
        logdist (function for calculating log probability, e.g. log_pstar)
    out:
      array samples, (D, N) samples stored in columns
    """
    
    D = x.shape[0]
    samples = np.zeros((D, N))
    log_Px = logdist(x, *args)
    if not isinstance(widths, np.ndarray):
        widths = np.repeat([widths], D)
    
    for sample_idx in range(0, N + burn):
        print(f'\rIteration {sample_idx - burn}   ', end='')
        log_uprime = np.log(np.random.rand()) + log_Px
        
        # for each axis
        for axis in range(0, D):
            x_l = x.copy()
            x_r = x.copy()
            xprime = x.copy()
            
            # create a horizontal interval (x_l, x_r) enclosing xx
            random_amount = np.random.rand()
            x_l[axis] = x[axis] - random_amount * widths[axis]
            x_r[axis] = x[axis] + (1 - random_amount) * widths[axis]
            
            if step_out:
                while (logdist(x_l, *args) > log_uprime).all():
                    x_l[axis] -= widths[axis]
                while (logdist(x_r, *args) > log_uprime).all():
                    x_r[axis] += widths[axis]

            # propose xprimes and shrink interval until a good one is found
            step = 0
            while True:
                step += 1
                # print(f'\rIteration {sample_idx - burn}, step {step}   ',
                #       end='')

                xprime[axis] = (np.random.rand() * (x_r[axis] - x_l[axis])
                                + x_l[axis]).copy()
                log_Px = logdist(xprime, *args)
                
                if log_Px > log_uprime:
                    # only way to leave the loop; has to eventually be true
                    break
                else:
                    # shrink
                    if xprime[axis] > x[axis]:
                        x_r[axis] = xprime[axis]
                    elif xprime[axis] < x[axis]:
                        x_l[axis] = xprime[axis]
                    else:
                        raise RuntimeError('Bug detected: shrunk to current '
                                           'position and still not acceptable')
            x[axis] = xprime[axis]
        
        # record the samples
        if sample_idx > burn:
            samples[:, sample_idx - burn] = x.squeeze()
    print('')
    return samples


def dumb_metropolis(init, log_ptilde, iters, sigma, *args):
    """
    Code adapted from Iain Murray's matlab code.
    
    in:
      scalar:
        iters (number of samples)
        sigma (step size for proposals)
        *args (other arguments passed to log_ptilde)
      vector:
        init (initial condition; numpy array)
      function:
        log_ptilde (function that takes inputs and returns log probability of
                    the target density up to a constant)
    out:
      array samples, (D, iters) states visited by the markov chain
      scalar arate, the acceptance rate (should be a little bit less than 0.5;
             tune sigma)
    """
    
    D = init.shape[0]
    samples = np.zeros(shape=(D, iters))
    
    arate = 0
    state = init
    Lp_state = log_ptilde(state, *args)
    
    for ss in range(0, iters):
        # propose
        prop = state + sigma * np.random.randn(*state.shape)
        Lp_prop = log_ptilde(prop, *args)
        
        if (np.log(np.random.rand()) < (Lp_prop - Lp_state)).all():
            # accept
            arate = arate + 1
            state = prop
            Lp_state = Lp_prop
        
        samples[:, ss] = state.squeeze()
    arate = arate / iters
    return (samples, arate)
