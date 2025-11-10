
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def mad_robust(x):   # Robust scale
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def compute_log_returns_from_prices(df, price_col="Adj Close"):
    px = df[price_col].dropna().astype(float).values
    r = np.diff(np.log(px)) * 100.0  # percent
    return r

def loglik_t(theta, r):
    mu, log_sigma, eta = theta
    sigma = np.exp(log_sigma)
    nu = 2.0 + np.exp(eta)
    return np.sum(stats.t.logpdf(r, df=nu, loc=mu, scale=sigma))

def logprior(theta, tau_mu, c_sigma, mean_nu_star):
    mu, log_sigma, eta = theta
    sigma = np.exp(log_sigma)
    nu_star = np.exp(eta)
    lp = stats.norm.logpdf(mu, 0.0, tau_mu)
    lp += np.log(2.0) - np.log(np.pi * c_sigma) - np.log(1.0 + (sigma / c_sigma)**2)
    rate = 1.0 / mean_nu_star
    lp += np.log(rate) - rate * nu_star
    lp += log_sigma + eta  # Jacobian
    return lp

def logpost(theta, r, tau_mu, c_sigma, mean_nu_star):
    return loglik_t(theta, r) + logprior(theta, tau_mu, c_sigma, mean_nu_star)

def mh_sampler(r, n_iter=15000, burn_frac=0.5, step_mu=0.05, step_logsig=0.05, step_eta=0.05,
               tau_mu=0.5, c_sigma=1.5, mean_nu_star=30.0, init=None, random_seed=42):
    rng = np.random.default_rng(random_seed)
    if init is None:
        s_robust = mad_robust(r)
        mu0 = 0.0
        logs0 = np.log(max(s_robust, 1e-6))
        eta0 = np.log(8.0 - 2.0)   # nu0 ~ 8
        theta = np.array([mu0, logs0, eta0], dtype=float)
    else:
        theta = np.array(init, dtype=float)

    steps = np.array([step_mu, step_logsig, step_eta], dtype=float)
    acc = np.zeros(3, dtype=int)
    chain = np.zeros((n_iter, 3), dtype=float)
    lp_cur = logpost(theta, r, tau_mu, c_sigma, mean_nu_star)

    for it in range(n_iter):
        for j in range(3):
            prop = theta.copy()
            prop[j] += rng.normal(0.0, steps[j])
            lp_prop = logpost(prop, r, tau_mu, c_sigma, mean_nu_star)
            if np.log(rng.random()) <= (lp_prop - lp_cur):
                theta = prop
                lp_cur = lp_prop
                acc[j] += 1
        chain[it, :] = theta

    burn = int(burn_frac * n_iter)
    post = chain[burn:, :]
    acc_rate = acc / n_iter
    return post, acc_rate, chain

def plugin_var_es(r, post, alphas=(0.95, 0.99)):
    mu_med = np.median(post[:, 0])
    sig_med = np.exp(np.median(post[:, 1]))
    nu_med = 2.0 + np.exp(np.median(post[:, 2]))
    out = []
    for a in alphas:
        q = stats.t.ppf(1.0 - a, df=nu_med, loc=mu_med, scale=sig_med)
        var_a = -q
        rng = np.random.default_rng(123)
        z = rng.standard_t(df=nu_med, size=200000) * sig_med + mu_med
        L = -z
        VaR = np.quantile(L, a)
        ES = L[L >= VaR].mean()
        out.append((a, var_a, ES))
    return out, (mu_med, sig_med, nu_med)

def quick_plots(r, mu_hat, sig_hat, nu_hat, out_prefix="t_fit"):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.hist(r, bins=50, density=True, alpha=0.6)
    xs = np.linspace(np.min(r), np.max(r), 400)
    ys = stats.t.pdf(xs, df=nu_hat, loc=mu_hat, scale=sig_hat)
    ax1.plot(xs, ys)
    ax1.set_title("Returns histogram with fitted t density")
    fig1.tight_layout()
    fig1.savefig(out_prefix + "_hist.png", dpi=144)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    percs = np.linspace(0.01, 0.99, 99)
    q_emp = np.quantile(r, percs)
    q_the = stats.t.ppf(percs, df=nu_hat, loc=mu_hat, scale=sig_hat)
    ax2.scatter(q_the, q_emp, s=10)
    lo = min(q_the.min(), q_emp.min())
    hi = max(q_the.max(), q_emp.max())
    ax2.plot([lo, hi], [lo, hi])
    ax2.set_title("QQ-plot: empirical vs fitted t")
    fig2.tight_layout()
    fig2.savefig(out_prefix + "_qq.png", dpi=144)

def main():
    csv_path = "/mnt/data/prices.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        r = compute_log_returns_from_prices(df, "Adj Close")
        src = "Loaded /mnt/data/prices.csv"
    else:
        rng = np.random.default_rng(0)
        mu_true, sig_true, nu_true = 0.02, 1.0, 8.0
        r = rng.standard_t(df=nu_true, size=2500) * sig_true + mu_true
        src = "Synthetic demo (no prices.csv found)"

    s_rob = mad_robust(r)
    tau_mu = float(min(0.50, 0.25 * s_rob))
    c_sigma = float(2.0 * s_rob)
    mean_nu_star = 30.0

    post, acc_rate, chain = mh_sampler(
        r,
        n_iter=15000,
        burn_frac=0.5,
        step_mu=0.05,
        step_logsig=0.05,
        step_eta=0.05,
        tau_mu=tau_mu,
        c_sigma=c_sigma,
        mean_nu_star=mean_nu_star,
        init=None,
        random_seed=42
    )

    mu_med = float(np.median(post[:,0]))
    sig_med = float(np.exp(np.median(post[:,1])))
    nu_med = float(2.0 + np.exp(np.median(post[:,2])))

    risk, params = plugin_var_es(r, post, alphas=(0.95, 0.99))

    quick_plots(r, mu_med, sig_med, nu_med, out_prefix="/mnt/data/t_fit")

    summ = pd.DataFrame({
        "param": ["mu(%)","sigma(%)","nu"],
        "median":[mu_med, sig_med, nu_med],
        "p2.5":[np.percentile(post[:,0],2.5),
                np.percentile(np.exp(post[:,1]),2.5),
                2.0 + np.percentile(np.exp(post[:,2]),2.5)],
        "p97.5":[np.percentile(post[:,0],97.5),
                 np.percentile(np.exp(post[:,1]),97.5),
                 2.0 + np.percentile(np.exp(post[:,2]),97.5)]
    })

    risk_df = pd.DataFrame([{"alpha": a, "VaR(%)": v, "ES(%)": e} for (a,v,e) in risk])

    print("=== Data source ===")
    print(src)
    print("\n=== Prior hyper-parameters (data-adaptive) ===")
    print(f"tau_mu = {tau_mu:.4f} (%),  c_sigma = {c_sigma:.4f} (%),  mean_nu_star = {mean_nu_star:.1f}")
    print("\n=== Acceptance rates (mu, log_sigma, eta) ===")
    print(acc_rate)
    print("\n=== Posterior summaries ===")
    print(summ.to_string(index=False))
    print("\n=== Plug-in risk metrics ===")
    print(risk_df.to_string(index=False))
    print("\nSaved plots to /mnt/data/t_fit_hist.png and /mnt/data/t_fit_qq.png")

    summ.to_csv("/mnt/data/posterior_summary.csv", index=False)
    risk_df.to_csv("/mnt/data/risk_plugin.csv", index=False)

if __name__ == "__main__":
    main()
