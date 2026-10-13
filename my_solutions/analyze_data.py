def analyse_data(data,
                 log_posterior_fn, model_fn, predict_fn, ppd_test_statistic_fn=None,
                 param_names=None,
                 derived_parameter_fn=None, derived_parameter_names=[],
                 derived_parameter_true=[],
                 theta_true=None,
                 theta_init=None, theta_init_std=None,
                 plot=True, model_x=None,
                 n_step_emcee=5000, n_ppd=500):
    """
    Implement a Bayesian workflow on 1-dimensional data that contains
    an independent variable (x), a dependent variable (y) and uncertainty
    on the measured dependent variable (y_err). 

    It first calculaes a maximum a posteriori value for the parameters using scipy.optimize.
    
    Then it uses Markov chain Monte Carlo to produce samples distributed according
    to the posterior distribution on the parameters of the model (using the 
    emcee package). 

    Subsequently, it calculates model predictive and posterior predictive distributions,
    plots them, and performs a posterior predictive check if there is a function
    that is supplied to do so (`ppd_test_statistic`, discussed in detail Week 7).

    Parameters:
    -----------
    log_posterior_fn : Callable function
        A callable function that takes in (theta, x, y_err, y) where
        theta is the list of free parameters.
    model_fn : Callable function
        A callable function that takes in (theta, x) as parameters
        and returns an estimate of y. 
    predict_fn : Callable function
        A callable function that takes in (theta, x, y_err, y) and produces
        predicted data from that model (simulates data from the likelihood, given
        a set of parameters).
    ppd_test_statistic_function : Callable function
        Takes in (y, theta, x, sigma_y) and produces a single summary statistic
        that can be used to compare replicated (posterior predictive) data with real data.
    derived_parameter_fn : Callable function
        If there are interesting parameters that are deterministically derived from the
        parameters that we sample, then they are defined here.
    derived_parameter_names : list
        A list of the names of the derived parameters output from `derived_parameter_fn`
    derived_parameter_true : list
        A list of the *true* derived parameters (if known)
    theta_true : np.ndarray, or list
        A list of the true parameters that were sampled (if known)
    theta_init : np.ndarray
        The initial values for the sampled parameters at the start of MCMC sampling
    theta_std : np.ndarray
        We can initialize the sampler around theta_init with some standard deviation.
    plot : bool
        True to make summary plots. False to not make them. 
    n_step_emcee : int, default=5000
        number of MCMC steps to take
    n_ppd : int, default=500
        number of posterior samples on which to calculate ppd_test_statistic

    Returns:
    --------
    results : dict
        A results dictionary containing samples, true values, posterior fit information
        and (if supplied) posterior predictive checking information.
    """

    x, y, sigma_y = data["x"], data["y"], data["y_err"]

    # MCMC initialization
    if theta_init is None:
        theta_init = theta_true
    theta_init = np.array(theta_init)

    if theta_init_std is None:
        theta_init_std = 0.1*theta_init
        theta_init_std[theta_init_std == 0.0] = 0.1

    # names of parameters
    if param_names is None:
        param_names = [f"p_{i}" for i in range(theta_init.size)]

    if model_x is None:
        model_x = x

    if theta_true is not None:
        true_model_plot_def = [dict(
            x=model_x, y=model_fn(theta_true, model_x),
            style=dict(color="black", ls="--", label="True model")
        )]
    else:
        true_model_plot_def = []

    # Find the MAP
    def negative_log_posterior(theta, x, sigma_y, y):
        return -log_posterior_fn(theta, x, sigma_y, y)

    MAP_result = scipy.optimize.minimize(
        fun=negative_log_posterior,
        x0=theta_init,
        args=(x, sigma_y, y)
    )
    theta_MAP = MAP_result.x

    print("MAP results")
    for name, theta in zip(param_names, theta_MAP):
        print(f"{name}_MAP = {theta}")


    # Sample posterior with emcee

    # emcee requires some extra settings to run
    n_param = len(theta_init) # Number of parameter we are sampling
    n_walker = 3*n_param      # Number of walkers. This just needs to be larger than 2*n_param + 1
    n_step = n_step_emcee     # How many steps each walker will take. The number of samples will be n_walker*n_step

    # The starting point for each walker
    theta_init = theta_init + theta_init_std*np.random.normal(size=(n_walker, n_param))

    # compute derived parameters while sampling
    if derived_parameter_fn is not None:
        log_posterior_wrapper = lambda *args: (log_posterior_fn(*args), derived_parameter_fn(args[0]))
    else:
        log_posterior_wrapper = log_posterior_fn
    sampler = emcee.EnsembleSampler(
        nwalkers=n_walker, ndim=n_param,
        log_prob_fn=log_posterior_wrapper,
        args=(x, sigma_y, y)
    )
    state = sampler.run_mcmc(theta_init, nsteps=n_step, progress=True)

    # The samples will be correlated, this checks how correlated they are
    # We will discuss this once we come to MCMC methods
    integrated_autocorr_time = sampler.get_autocorr_time(quiet=True)
    print("Auto-correlation time of chain:")
    for name, value in zip(param_names, integrated_autocorr_time):
        print(f"{name} = {value:.1f}")

    max_autocorr_time = max(integrated_autocorr_time)

    # We need to discard the beginning of the chain (a few auto-correlation times)
    # to get rid of the initial conditions
    burn_in_config = dict(
        discard=int(5*max_autocorr_time),
        thin=int(max_autocorr_time/2),
        flat=True
    )
    chain = sampler.get_chain(
        **burn_in_config
    )

    # Make predictive distributions
    # Choose a small subsample of the chain for plotting purposes
    chain_samples = chain[np.random.choice(chain.shape[0], size=n_ppd)]
    # Evaluate the model at the sample parameters
    model_predictive = np.array(
        [model_fn(sample, model_x) for sample in chain_samples]
    )
    model_quantiles = np.quantile(
        model_predictive, q=[0.025, 0.16, 0.84, 0.975], axis=0
    )

    posterior_predictive = np.array(
        [predict_fn(sample, x, sigma_y) for sample in chain_samples]
    )
    posterior_predictive_quantiles = np.quantile(
        posterior_predictive, q=[0.025, 0.16, 0.84, 0.975], axis=0
    )

    if ppd_test_statistic_fn is not None:
        T_rep = np.array(
            [ppd_test_statistic_fn(y_rep, sample, x, sigma_y)
             for y_rep, sample in zip(posterior_predictive, chain_samples)]
        )
        T_data = np.array(
            [ppd_test_statistic_fn(y, sample, x, sigma_y)
             for sample in chain_samples]
        )
        goodness_of_fit_pte = np.sum((T_rep - T_data) > 0)/len(T_rep)

        print(f"Posterior predictive p-value: {goodness_of_fit_pte:.2g} "
              f"(T_rep = {np.mean(T_rep):.2g}±{np.std(T_rep):.2g}, "
              f"T_data = {np.mean(T_data):.2g}±{np.std(T_data):.2g})"
        )

    if derived_parameter_fn is not None:
        blobs = sampler.get_blobs(
            **burn_in_config
        )
        if blobs.ndim == 1:
            blobs = blobs.reshape(-1, 1)
        chain = np.concatenate((chain, blobs), axis=-1)

    print("Posterior results (mean±std)")
    for i, name in enumerate(param_names + derived_parameter_names):
        print(f"{name} = {np.mean(chain[:,i]):.2g}±{np.std(chain[:,i]):.2g}")

    if plot:
        # Make a corner plot
        fig = plt.figure()
        fig = corner.corner(
            chain,
            bins=40,
            labels=param_names + derived_parameter_names,
            truths=theta_true + derived_parameter_true,
            levels=1-np.exp(-0.5*np.array([1, 2])**2), # Credible contours corresponding to 1 and 2 sigma in 2D
            quantiles=[0.025, 0.16, 0.84, 0.975],
            fig=fig
        )
        plot_data(
            x=x, y=y, y_err=sigma_y,
            models=[
                dict(x=model_x, y=model_fn(theta_MAP, model_x),
                    style=dict(color="C2", label="MAP model")),
                
                dict(x=model_x, lower=model_quantiles[0], upper=model_quantiles[-1],
                    style=dict(color="C1", alpha=0.5, label="Model predictions")),
                dict(x=model_x, lower=model_quantiles[1], upper=model_quantiles[-2],
                    style=dict(color="C1", alpha=0.5)),
                dict(x=x, violin_samples=posterior_predictive,
                     style=dict(color="grey", alpha=0.5, label="Posterior predictions")),
            ] + true_model_plot_def,
            marker=".", linestyle="none"
        )
        # plot our posterior predictive test statistic
        if ppd_test_statistic_fn is not None:
            fig = plt.figure(figsize=(6, 6))
            gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 2],
                                  hspace=0.05, wspace=0.05)
            ax = fig.add_subplot(gs[1, 0])
            ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
        
            lims = [min(T_data.min(), T_rep.min()), max(T_data.max(), T_rep.max())]
            # edges = np.linspace(*lims, bins + 1)
        
            ax.scatter(T_data, T_rep, s=5, alpha=0.5)
            ax.plot(lims, lims, "k--", lw=1, label="$y = x$")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel(r"$T(y, \theta)$")
            ax.set_ylabel(r"$T(y_\mathrm{rep}, \theta)$")
            ax.legend(frameon=False, loc="upper left")
        
            ax_top.hist(T_data, bins='auto', alpha=0.5)
            ax_right.hist(T_rep, bins='auto', alpha=0.5, orientation="horizontal")
            ax_top.tick_params(labelbottom=False)
            ax_right.tick_params(labelleft=False)
            ax_top.set_title(f"$p = {goodness_of_fit_pte:.2g}$")
            

    return dict(
        MAP=theta_MAP,
        PPD=posterior_predictive,
        TPD=model_predictive,
        PPD_params=chain_samples,
        chain=chain
    )