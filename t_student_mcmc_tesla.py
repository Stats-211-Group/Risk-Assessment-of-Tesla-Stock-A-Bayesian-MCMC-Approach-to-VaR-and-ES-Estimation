import os
import pandas as pd
import numpy as np
from scipy import stats

# ===================================================================
# 1) Global Configuration
# ===================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Tesla_close.csv")  # Data file path
RETURNS_ARE_PERCENT = False        # Whether returns in CSV are already in percentage

OUT_DIR = os.path.join(BASE_DIR, "outputs")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

REPORT_FILENAME = "analysis_report.txt"

N_ITER    = 100000  # Total MCMC iterations
BURN_FRAC = 0.1    # Burn-in fraction

# MH step size (needs adjustment based on acceptance rate)(percentage of the total N_iter)
STEP_MU   = 0.30 # Slightly larger step to lower acceptance rate and boost ESS
STEP_ETA  = 0.15

SEED = 42 # Random seed for reproducibility

# ===================================================================

# ============ Utility Helpers ============

def load_returns(csv_path, returns_are_percent=True):
    """Load return series from the CSV file."""
    df = pd.read_csv(csv_path)
    r = df['Log_Return'].dropna().values
    if not returns_are_percent:
        r = r * 100.0
    return r

# ---------- Priors ----------
def logprior(theta, tau_mu, alpha_ig, beta_ig, mean_nu_star):
    """
        log prior:
            mu ~ N(0, tau_mu^2)
            sigma^2 ~ Inverse-Gamma(alpha_ig, beta_ig)
            nu ~ Exponential(mean = mean_nu_star)
        Note: the code works in (log_sigma, eta) space, so include Jacobian +log_sigma + eta
    """
    mu, log_sigma, eta = theta
    sigma   = np.exp(log_sigma)      # sigma > 0
    sigma2  = sigma ** 2
    nu = np.exp(eta)                 # nu > 0

    # Normal prior for mu
    lp = stats.norm.logpdf(mu, 0.0, tau_mu)

    # Inverse-gamma prior for sigma^2 (log-density)
    lp += stats.invgamma.logpdf(sigma2, a=alpha_ig, scale=beta_ig)

    # Exponential prior for nu (mean mean_nu_star; rate = 1/mean)
    rate = 1.0 / mean_nu_star
    lp += np.log(rate) - rate * nu

    # Jacobian terms: sigma = exp(log_sigma) → +log_sigma; nu = exp(eta) → +eta
    lp += log_sigma + eta
    return lp

# ---------- Likelihood in the hierarchical form ----------
def loglik_normal_given_lambda(theta, r, lam):
    """
    Conditional log-likelihood given the latent λ_t:
        r_t | λ_t ~ Normal(mu, sigma^2 / λ_t)
    Here theta = (mu, log_sigma, eta); eta only affects λ's conditional draw and is not used directly.
    """
    mu, log_sigma, _ = theta
    sigma = np.exp(log_sigma)  # Actual scale
    # For each t: variance = sigma^2 / lam_t
    var_t = (sigma ** 2) / lam
    # Sum the normal log-density term-by-term
    # log N(r | mu, var_t) = -0.5*log(2π) - 0.5*log(var_t) - (r-mu)^2 / (2*var_t)
    ll = -0.5 * np.log(2.0 * np.pi) * len(r)
    ll += -0.5 * np.sum(np.log(var_t))
    ll += -0.5 * np.sum(((r - mu) ** 2) / var_t)
    return ll

def loglik_lambda_given_nu(lam, nu):
    """
    Compute log P(lambda | nu)
    lambda_t ~ Gamma(shape=nu/2, rate=nu/2)
    """
    shape = nu / 2.0
    rate  = nu / 2.0 # scipy notation: scale = 1/rate
    # Use scipy's logpdf and sum
    return np.sum(stats.gamma.logpdf(lam, a=shape, scale=1.0/rate))

def logpost_hier(theta, r, lam, tau_mu, alpha_ig, beta_ig, mean_nu_star):
    mu, log_sigma, eta = theta
    nu = np.exp(eta)
    
    # 1. log P(r | lam, mu, sigma)
    ll_r = loglik_normal_given_lambda(theta, r, lam)
    
    # 2. log P(theta)
    lp_theta = logprior(theta, tau_mu, alpha_ig, beta_ig, mean_nu_star)
    
    # 3. log P(lam | nu)
    ll_lam = loglik_lambda_given_nu(lam, nu)
    
    return ll_r + lp_theta + ll_lam

# ---------- Gibbs update for λ_t ----------
def gibbs_sample_lambda(r, mu, sigma, nu, rng):
    """
    Perform a Gibbs update for the entire λ_{1:n} vector:
        λ_t | r_t, mu, sigma, nu ~ Gamma((nu+1)/2, rate = (nu + ((r_t-mu)^2 / sigma^2)) / 2)
    SciPy uses gamma(shape=k, scale=θ) with θ = 1/rate.
    Returns lam with shape [n,].
    rng: numpy.random.Generator instance for reproducibility.
    """
    n = len(r)
    shape = (nu + 1.0) / 2.0
    # rate_t = ( nu + ((r_t - mu)^2 / sigma^2) ) / 2
    rate_t = (nu + ((r - mu) ** 2) / (sigma ** 2)) / 2.0
    scale_t = 1.0 / rate_t  # SciPy requires scale = 1/rate

    # Sample independently for each t (vectorized)
    # scipy.stats.gamma supports vectorized scales
    lam = stats.gamma.rvs(a=shape, scale=scale_t, size=n, random_state=rng)
    return lam

# ============ Metropolis-within-Gibbs Sampler (minimal modification version) ============

def mwg_sampler(r,
                n_iter, burn_frac,
                step_mu, step_eta,
                tau_mu, alpha_ig, beta_ig, mean_nu_star,
                random_seed,
                init=None, thin=50):
    """
        Metropolis-within-Gibbs (MwG):
            1) Gibbs: sample the entire λ vector given (mu, sigma, nu)
            2) Gibbs: draw sigma^2 directly via the inverse-gamma conjugate
            3) MH: propose random-walk updates for (mu, eta)
        Unlike a pure MH sampler, both lambda and sigma^2 have Gibbs steps.
    """
    rng = np.random.default_rng(random_seed)

    # ------ Initialize θ and λ ------
    if init is None:
        mu0   = 0.0
        logs0 = np.log(np.std(r, ddof=1)) # Use the sample standard deviation as the starting scale
        eta0  = np.log(8.0)   # Initialize nu ≈ 8
        theta = np.array([mu0, logs0, eta0], dtype=float)
    else:
        theta = np.array(init, dtype=float)

    # Initialize λ: run one Gibbs step using the current parameters (or start from all ones)
    mu, log_sigma, eta = theta
    sigma = np.exp(log_sigma)
    nu    = np.exp(eta)
    lam   = gibbs_sample_lambda(r, mu, sigma, nu, rng)

    mh_indices = (0, 2)  # indices for mu and eta within theta
    mh_steps   = np.array([step_mu, step_eta], dtype=float)
    acc        = np.zeros(len(mh_indices), dtype=int)
    chain   = np.zeros((n_iter, 3), dtype=float)  # Store the θ trajectory
    lp_cur  = logpost_hier(theta, r, lam, tau_mu, alpha_ig, beta_ig, mean_nu_star)

    # ------ Main loop ------
    for it in range(n_iter):
        # (A) Gibbs update for the full λ vector
        mu, log_sigma, eta = theta
        sigma = np.exp(log_sigma)
        nu    = np.exp(eta)
        lam   = gibbs_sample_lambda(r, mu, sigma, nu, rng)

        # (B) Gibbs update for sigma^2 via the IG conjugate
        resid = r - mu
        shape_post = alpha_ig + len(r) / 2.0
        scale_post = beta_ig + 0.5 * np.sum(lam * (resid ** 2))
        sigma2 = stats.invgamma.rvs(a=shape_post, scale=scale_post, random_state=rng)
        log_sigma = 0.5 * np.log(sigma2)
        theta[1] = log_sigma

        # Update log posterior for the MH step
        lp_cur = logpost_hier(theta, r, lam, tau_mu, alpha_ig, beta_ig, mean_nu_star)

        # (C) MH updates for mu and eta
        for idx, param_idx in enumerate(mh_indices):
            prop = theta.copy()
            prop[param_idx] += rng.normal(0.0, mh_steps[idx])  # Symmetric normal proposal

            # Compute the proposed log posterior (likelihood conditions on lam)
            lp_prop = logpost_hier(prop, r, lam, tau_mu, alpha_ig, beta_ig, mean_nu_star)

            # MH rule: accept if log(u) <= logpost_prop - logpost_cur
            if np.log(rng.random()) <= (lp_prop - lp_cur):
                theta  = prop
                lp_cur = lp_prop
                acc[idx] += 1

        chain[it, :] = theta

    # ------ Return posterior draws and acceptance rates ------
    burn     = int(burn_frac * n_iter)
    # Thinning for better ACF
    post     = chain[burn::thin, :]
    acc_rate = np.array([acc[0] / n_iter, 1.0, acc[1] / n_iter])
    return post, acc_rate, chain

def calculate_var_es(mu_samps, sigma_samps, nu_samps, alpha=0.05):
    """
    Compute Value at Risk (VaR) and Expected Shortfall (ES)
    using the analytic formulas of the Student-t distribution.

    Parameters:
    alpha: Tail probability (default 0.05 for 95% confidence).
        Left tail is considered (losses).

    Returns:
    var_samps: Posterior draws of VaR (typically negative).
    es_samps:  Posterior draws of ES (more negative; mean loss beyond VaR).
    """
    n = len(mu_samps)
    var_samps = np.zeros(n)
    es_samps = np.zeros(n)
    
    # Iterate over posterior draws
    for i in range(n):
        mu  = mu_samps[i]
        sig = sigma_samps[i]
        nu  = nu_samps[i]
        
        # 1. VaR
        # VaR_alpha equals the alpha quantile of the t distribution
        q_std = stats.t.ppf(alpha, df=nu)
        var_samps[i] = mu + sig * q_std
        
        # 2. ES = E[R | R < VaR]
        # For X ~ t(nu, mu, sigma^2): ES = mu + sigma * ES_standard(nu, alpha)
        # ES_standard = - ((nu + q^2) / (nu - 1)) * (pdf(q) / alpha)
        
        if nu <= 1:
            # Expectation does not exist
            es_samps[i] = -np.inf 
        else:
            pdf_q = stats.t.pdf(q_std, df=nu)
            es_std_term = - ((nu + q_std**2) / (nu - 1)) * (pdf_q / alpha)
            es_samps[i] = mu + sig * es_std_term
            
    return var_samps, es_samps


# --------------------------
# 5) Plotting helpers
# --------------------------
import matplotlib.pyplot as plt

def plot_hist(x, title, xlabel, outpath):
    """Beautified histogram with an overlaid KDE curve."""
    with plt.style.context('seaborn-v0_8-darkgrid'):
        fig = plt.figure(figsize=(10, 6))
        ax  = fig.add_subplot(111)
        # Histogram
        ax.hist(x, bins=50, density=True, alpha=0.7, label="Posterior samples histogram")
        # KDE overlay
        kde = stats.gaussian_kde(x)
        x_range = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_range, kde(x_range), color='firebrick', linewidth=2, label="KDE estimate")
        ax.set_title(title, fontsize=16, weight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outpath, dpi=144)
        plt.close(fig)

def plot_trace(x, title, ylabel, outpath):
    """Beautified MCMC trace plot with its mean highlighted."""
    with plt.style.context('seaborn-v0_8-darkgrid'):
        fig = plt.figure(figsize=(12, 6))
        ax  = fig.add_subplot(111)
        ax.plot(x, alpha=0.8, label="MCMC trace")
        # Mean reference line
        mean_val = np.mean(x)
        ax.axhline(mean_val, color='firebrick', linestyle='--', linewidth=2, label=f"Mean: {mean_val:.3f}")
        ax.set_title(title, fontsize=16, weight='bold')
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outpath, dpi=144)
        plt.close(fig)

def quick_fit_plots(r, mu_hat, sig_hat, nu_hat, out_prefix):
    """Produce quick-fit visuals: histogram + fitted t density and QQ-plot."""
    with plt.style.context('seaborn-v0_8-darkgrid'):
        # 1. Histogram + fitted t density
        fig1 = plt.figure(figsize=(10, 6))
        ax1  = fig1.add_subplot(111)
        ax1.hist(r, bins=50, density=True, alpha=0.7, label="Return histogram")
        xs = np.linspace(np.min(r), np.max(r), 400)
        ys = stats.t.pdf(xs, df=nu_hat, loc=mu_hat, scale=sig_hat)
        ax1.plot(xs, ys, color='firebrick', linewidth=2, label=f"Fitted t density (ν={nu_hat:.2f})")
        ax1.set_title("Return histogram with fitted t density", fontsize=16, weight='bold')
        ax1.set_xlabel("Daily return (%)", fontsize=12)
        ax1.set_ylabel("Density", fontsize=12)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig(out_prefix + "_hist.png", dpi=144)
        plt.close(fig1)

        # 2. QQ-plot (empirical quantiles vs fitted t quantiles)
        fig2 = plt.figure(figsize=(8, 8))
        ax2  = fig2.add_subplot(111)
        # stats.probplot makes the QQ plot straightforward
        stats.probplot(r, dist=stats.t, sparams=(nu_hat, mu_hat, sig_hat), plot=ax2)
        # Cosmetic tweaks
        ax2.get_lines()[0].set_markerfacecolor('deepskyblue')
        ax2.get_lines()[0].set_markeredgecolor('deepskyblue')
        ax2.get_lines()[0].set_alpha(0.6)
        ax2.get_lines()[1].set_color('firebrick')
        ax2.set_title("QQ plot: empirical vs fitted t quantiles", fontsize=16, weight='bold')
        ax2.set_xlabel("Theoretical quantiles", fontsize=12)
        ax2.set_ylabel("Sample quantiles", fontsize=12)
        fig2.tight_layout()
        fig2.savefig(out_prefix + "_qq.png", dpi=144)
        plt.close(fig2)

def acf_1d(x, max_lag=100):
    """
    Compute sample autocorrelation ρ(k) for k = 0..max_lag.
    ρ(0)=1 and the series is normalized by variance.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 2:
        return np.array([1.0])
    ac_full = np.correlate(x, x, mode='full')
    ac = ac_full[n-1:n+max_lag]
    denom = ac[0]
    if denom <= 0:
        return np.zeros_like(ac)
    acf = ac / denom
    return acf

def plot_acf_series(x, title, outpath, max_lag=100):
    """
    Beautified ACF plot with an added confidence band.
    """
    n = len(x)
    rho = acf_1d(x, max_lag=max_lag)
    with plt.style.context('seaborn-v0_8-darkgrid'):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        lags = np.arange(len(rho))
        # 95% confidence band (±1.96/sqrt(n))
        conf_level = 1.96 / np.sqrt(n)
        ax.fill_between(lags, -conf_level, conf_level, color='lightblue', alpha=0.5, label="95% confidence band")
        # Lollipop-style ACF bars
        ax.stem(lags, rho, linefmt='gray', markerfmt='o', basefmt=' ')
        ax.set_title(f"Autocorrelation (ACF): {title}", fontsize=16, weight='bold')
        ax.set_xlabel("Lag k", fontsize=12)
        ax.set_ylabel("Autocorrelation ρ(k)", fontsize=12)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outpath, dpi=144)
        plt.close(fig)

def ess_from_acf(x, max_lag=200):
    """
    Simple ESS estimate from the ACF: ESS ≈ n / (1 + 2 * sum_{k>=1} ρ(k))
    Uses pairwise truncation (stop when ρ(k)+ρ(k+1) <= 0) to avoid noisy tails.
    """
    rho = acf_1d(x, max_lag=max_lag)
    n = len(x)
    s = 0.0
    for k in range(1, len(rho), 2):
        pair = rho[k] + (rho[k+1] if k+1 < len(rho) else 0.0)
        if pair <= 0:
            break
        s += pair
    tau = 1.0 + 2.0 * s        # Integrated autocorrelation time
    return n / max(tau, 1e-12) # Avoid division by zero

def geweke_diagnostic(chain, first=0.1, last=0.5):
    """
    Performs the Geweke diagnostic test.
    Compares the mean of the first 10% of the chain with the last 50%.
    Returns a Z-score. 
    |Z| > 1.96 indicates non-convergence (at 5% significance level).
    """
    # 1. Split the chain
    n = len(chain)
    n_first = int(n * first)
    n_last  = int(n * last)
    
    if n_first == 0 or n_last == 0:
        return np.nan
        
    data_first = chain[:n_first]
    data_last  = chain[-n_last:]
    
    # 2. Calculate Means
    mean_first = np.mean(data_first)
    mean_last  = np.mean(data_last)
    
    # 3. Calculate Variance (corrected for autocorrelation)
    # Approximation: Var(mean) = Var(x) / ESS
    # We use the ess_from_acf function you already have
    var_first = np.var(data_first, ddof=1) / ess_from_acf(data_first)
    var_last  = np.var(data_last, ddof=1) / ess_from_acf(data_last)
    
    # 4. Calculate Z-score
    z_score = (mean_first - mean_last) / np.sqrt(var_first + var_last)
    
    return z_score



# --------------------------
# 6) Main routine
# --------------------------
def main():
    """Pipeline: load data -> run sampler -> analyze and plot."""
    # Load returns (percent units)
    r = load_returns(CSV_PATH, returns_are_percent=RETURNS_ARE_PERCENT)

    # Prior hyperparameters (fixed scales for simplicity)
    hyper_params = {
        "tau_mu":       100,      # Prior std for mu (aligned with data scale)
        "alpha_ig":     0.01,     # Weakly informative IG shape
        "beta_ig":      0.01,     # Fixed scale parameter
        "mean_nu_star": 30.0      # Prior mean for nu
    }

    # Run the sampler
    post, acc_rate, chain = mwg_sampler(r,
                                        n_iter=N_ITER,
                                        burn_frac=BURN_FRAC,
                                        step_mu=STEP_MU,
                                        step_eta=STEP_ETA,
                                        random_seed=SEED,
                                        **hyper_params)

    # Analyze and produce outputs
    analyze_and_plot_results(r=r,
                             post=post,
                             chain=chain,
                             acc_rate=acc_rate,
                             sampler_name="Metropolis-within-Gibbs",
                             hyper_params=hyper_params)

def analyze_and_plot_results(r, post, chain, acc_rate, sampler_name, hyper_params):
    """
    Post-process MCMC draws, run diagnostics, produce plots, and persist outputs.
    """
    # Convert posterior samples to actual parameter scales
    mu_samps    = post[:, 0]
    sigma_samps = np.exp(post[:, 1])
    nu_samps    = np.exp(post[:, 2])

    # Posterior medians for plug-in fit
    mu_med = float(np.median(mu_samps))
    sig_med = float(np.median(sigma_samps))
    nu_med = float(np.median(nu_samps))

    # Quick fit visuals (histogram + QQ)
    quick_fit_plots(r, mu_med, sig_med, nu_med,
                    out_prefix=os.path.join(OUT_DIR, "t_fit"))

    # ---- Posterior histograms ----
    plot_hist(mu_samps,    "Posterior of μ", "mu (%)",
              os.path.join(OUT_DIR, "post_mu_hist.png"))
    plot_hist(sigma_samps, "Posterior of σ", "sigma (%)",
              os.path.join(OUT_DIR, "post_sigma_hist.png"))
    plot_hist(nu_samps,    "Posterior of ν", "nu (df)",
              os.path.join(OUT_DIR, "post_nu_hist.png"))

    # ---- Trace plots using the full chain ----
    plot_trace(chain[:, 0], "Trace: μ", "mu (%)",
               os.path.join(OUT_DIR, "trace_mu.png"))
    plot_trace(np.exp(chain[:, 1]), "Trace: σ", "sigma (%)",
               os.path.join(OUT_DIR, "trace_sigma.png"))
    plot_trace(np.exp(chain[:, 2]), "Trace: ν", "nu (df)",
               os.path.join(OUT_DIR, "trace_nu.png"))
    
    # ---- ACF plots ----
    plot_acf_series(mu_samps,    "μ chain", os.path.join(OUT_DIR, "acf_mu.png"),    max_lag=100)
    plot_acf_series(sigma_samps, "σ chain", os.path.join(OUT_DIR, "acf_sigma.png"), max_lag=100)
    plot_acf_series(nu_samps,    "ν chain", os.path.join(OUT_DIR, "acf_nu.png"),    max_lag=100)

    # Summary table
    ess_vals = [
        ess_from_acf(mu_samps),
        ess_from_acf(sigma_samps),
        ess_from_acf(nu_samps),
    ]

    summary = pd.DataFrame({
        "param": ["mu(%)", "sigma(%)", "nu"],
        "median": [mu_med, sig_med, nu_med],
        "mean": [np.mean(mu_samps), np.mean(sigma_samps), np.mean(nu_samps)],
        "sd": [np.std(mu_samps, ddof=1),
                np.std(sigma_samps, ddof=1),
                np.std(nu_samps, ddof=1)],
        "p2.5": [np.percentile(mu_samps, 2.5),
                  np.percentile(sigma_samps, 2.5),
                  np.percentile(nu_samps, 2.5)],
        "p97.5": [np.percentile(mu_samps, 97.5),
                   np.percentile(sigma_samps, 97.5),
                   np.percentile(nu_samps, 97.5)],
        "ess": ess_vals,
    })
    summary_table = summary.to_string(index=False)
    
    # Calculate Geweke Z-scores
    z_mu = geweke_diagnostic(mu_samps)
    z_sigma = geweke_diagnostic(sigma_samps)
    z_nu = geweke_diagnostic(nu_samps)
    
    print("-" * 30)
    print("Geweke Diagnostics (Z-scores):")
    print(f"  Mu:    {z_mu:.3f}")
    print(f"  Sigma: {z_sigma:.3f}")
    print(f"  Nu:    {z_nu:.3f}")
    print("  (|Z| < 1.96 suggests convergence)")
    print("-" * 30)
        
    # ==========================================
    # Risk metrics (VaR & ES)
    # ==========================================
    # Tail probability, e.g., 5% (0.05) or 1% (0.01)
    ALPHA = 0.05 
    
    print(f"\nCalculating VaR and ES at alpha={ALPHA}...")
    var_chain, es_chain = calculate_var_es(mu_samps, sigma_samps, nu_samps, alpha=ALPHA)
    
    # Filter out undefined ES samples when nu <= 1
    valid_mask = np.isfinite(es_chain)
    invalid_es_count = int(np.sum(~valid_mask))
    if invalid_es_count > 0:
        print(f"Warning: {invalid_es_count} samples had nu <= 1 (ES undefined).")
        var_chain = var_chain[valid_mask]
        es_chain  = es_chain[valid_mask]
    
    # Summary statistics
    var_mean = np.mean(var_chain)
    var_lower = np.percentile(var_chain, 2.5)
    var_upper = np.percentile(var_chain, 97.5)
    
    es_mean = np.mean(es_chain)
    es_lower = np.percentile(es_chain, 2.5)
    es_upper = np.percentile(es_chain, 97.5)
    
    # Console summary
    print("-" * 40)
    print(f"Risk Measures (alpha={ALPHA*100}%) Summary:")
    print("-" * 40)
    print(f"Value-at-Risk (VaR):")
    print(f"  Mean Estimate: {var_mean:.4f}")
    print(f"  95% CI:       [{var_lower:.4f}, {var_upper:.4f}]")
    print(f"\nExpected Shortfall (ES):")
    print(f"  Mean Estimate: {es_mean:.4f}")
    print(f"  95% CI:       [{es_lower:.4f}, {es_upper:.4f}]")
    print("-" * 40)
    
    print("Risk measures recorded in the analysis report.")
    
    # Optional: visualize posterior distributions of VaR and ES
    with plt.style.context('seaborn-v0_8-darkgrid'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # VaR histogram
        axes[0].hist(var_chain, bins=50, density=True, alpha=0.7, color='steelblue')
        axes[0].axvline(var_mean, color='red', linestyle='--', label=f'Mean: {var_mean:.3f}')
        axes[0].set_title(f"Posterior Distribution of VaR ({ALPHA*100}%)")
        axes[0].set_xlabel("Return Level")
        axes[0].legend()
        
        # ES histogram
        axes[1].hist(es_chain, bins=50, density=True, alpha=0.7, color='orange')
        axes[1].axvline(es_mean, color='red', linestyle='--', label=f'Mean: {es_mean:.3f}')
        axes[1].set_title(f"Posterior Distribution of ES ({ALPHA*100}%)")
        axes[1].set_xlabel("Return Level")
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "risk_posterior.png"), dpi=144)
        plt.close(fig)

    # Consolidated text report
    report_lines = []
    report_lines.append("=== Run Configuration ===")
    report_lines.append(f"Sampler: {sampler_name}")
    report_lines.append(f"Iterations: {len(chain)}, burn-in fraction: {BURN_FRAC}")
    report_lines.append("")
    report_lines.append("Hyperparameters:")
    for key, value in hyper_params.items():
        report_lines.append(f"  {key}: {value}")
    report_lines.append("")
    report_lines.append("Acceptance rates (mu, log_sigma, eta): " + ", ".join(f"{x:.3f}" for x in acc_rate))
    report_lines.append("ESS (mu, sigma, nu): " + ", ".join(f"{x:.1f}" for x in ess_vals))
    report_lines.append("")
    report_lines.append("Posterior summary table:")
    report_lines.extend(summary_table.splitlines())
    report_lines.append("")
    report_lines.append("Geweke diagnostics (Z-scores):")
    report_lines.append(f"  mu   : {z_mu:.3f}")
    report_lines.append(f"  sigma: {z_sigma:.3f}")
    report_lines.append(f"  nu   : {z_nu:.3f}")
    report_lines.append("(|Z| < 1.96 suggests convergence)")
    report_lines.append("")
    report_lines.append(f"VaR mean (alpha={ALPHA*100:.1f}%): {var_mean:.4f}")
    report_lines.append(f"VaR 95% CI: [{var_lower:.4f}, {var_upper:.4f}]")
    report_lines.append(f"ES mean  (alpha={ALPHA*100:.1f}%): {es_mean:.4f}")
    report_lines.append(f"ES 95% CI: [{es_lower:.4f}, {es_upper:.4f}]")
    if invalid_es_count > 0:
        report_lines.append(f"Warning: {invalid_es_count} samples discarded because nu <= 1 made ES undefined.")

    report_path = os.path.join(OUT_DIR, REPORT_FILENAME)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))

    print("Posterior summaries and diagnostics saved to the outputs/ folder.")
    print(f"Detailed text report written to: {report_path}")


if __name__ == "__main__":
    main()