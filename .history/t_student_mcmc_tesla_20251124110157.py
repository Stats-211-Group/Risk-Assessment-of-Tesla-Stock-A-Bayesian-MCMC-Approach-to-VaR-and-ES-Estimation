

# --------------------------
# 5) 便捷图表函数
# --------------------------
def plot_hist(x, title, xlabel, outpath):
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.hist(x, bins=50, density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(outpath, dpi=144)
    plt.close(fig)

def plot_trace(x, title, ylabel, outpath):
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(x)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(outpath, dpi=144)
    plt.close(fig)

def quick_fit_plots(r, mu_hat, sig_hat, nu_hat, out_prefix):
    # 直方图 + 拟合 t 密度
    fig1 = plt.figure()
    ax1  = fig1.add_subplot(111)
    ax1.hist(r, bins=50, density=True, alpha=0.6)
    xs = np.linspace(np.min(r), np.max(r), 400)
    ys = stats.t.pdf(xs, df=nu_hat, loc=mu_hat, scale=sig_hat)
    ax1.plot(xs, ys)
    ax1.set_title("Returns histogram with fitted t density")
    fig1.tight_layout()
    fig1.savefig(out_prefix + "_hist.png", dpi=144)
    plt.close(fig1)

    # QQ-plot（经验分位数 vs 拟合 t 分位数）
    fig2 = plt.figure()
    ax2  = fig2.add_subplot(111)
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
    plt.close(fig2)
    
def acf_1d(x, max_lag=100):
    """
    计算样本自相关函数 ρ(k)，k=0..max_lag。
    ρ(0)=1。实现基于相关/方差的标准化。
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 2:
        return np.array([1.0])
    # 快速相关：full，再截取非负滞后部分
    ac_full = np.correlate(x, x, mode='full')
    ac = ac_full[n-1:n+max_lag]  # k=0..max_lag 的协方差（未除以 n）
    # 用 k=0 项做标准化，得到自相关 ρ(k)
    denom = ac[0]
    if denom <= 0:
        return np.zeros_like(ac)
    acf = ac / denom
    # 无偏近似（可选）：除以 (n-k)，但做相关时影响很小；这里省略
    return acf

def plot_acf_series(x, title, outpath, max_lag=100):
    """
    画 ACF 柱状图（k=0..max_lag）。lag=0 处为 1。
    """
    rho = acf_1d(x, max_lag=max_lag)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(range(len(rho)), rho, width=1.0)
    ax.set_title(f"ACF: {title}")
    ax.set_xlabel("Lag k")
    ax.set_ylabel("Autocorrelation ρ(k)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=144)
    plt.close(fig)

def ess_from_acf(x, max_lag=200):
    """
    基于 ACF 的简易 ESS 估计：ESS ≈ n / (1 + 2 * sum_{k>=1} ρ(k))
    使用成对截断（首次使 ρ(k)+ρ(k+1) <= 0 时停止累加），
    以避免噪声尾部导致过估计自相关。
    """
    rho = acf_1d(x, max_lag=max_lag)
    n = len(x)
    s = 0.0
    for k in range(1, len(rho), 2):
        pair = rho[k] + (rho[k+1] if k+1 < len(rho) else 0.0)
        if pair <= 0:
            break
        s += pair
    tau = 1.0 + 2.0 * s        # integrated autocorrelation time
    return n / max(tau, 1e-12) # 防止除零



# --------------------------
# 6) 主流程
# --------------------------
def main():
    # 读收益（单位：百分比）
    r = load_returns(CSV_PATH, returns_are_percent=RETURNS_ARE_PERCENT)

    # 用稳健尺度设先验超参（弱信息、与数据量级对齐）
    s_rob = mad_robust(r)
    tau_mu = float(min(0.50, 0.25 * s_rob))   # mu ~ N(0, tau_mu^2)
    c_sigma = float(2.0 * s_rob)              # sigma ~ Half-Cauchy(0, c_sigma)
    mean_nu_star = 30.0                       # (nu-2) ~ Exp(mean=30)

    # 采样
    post, acc_rate, chain = mh_sampler(
        r,
        n_iter=N_ITER,
        burn_frac=BURN_FRAC,
        step_mu=STEP_MU,
        step_logsig=STEP_LOGS,
        step_eta=STEP_ETA,
        tau_mu=tau_mu,
        c_sigma=c_sigma,
        mean_nu_star=mean_nu_star,
        init=None,
        random_seed=SEED
    )

    # 后验样本 → 真实参数
    mu_samps    = post[:, 0]
    sigma_samps = np.exp(post[:, 1])
    nu_samps    = 2.0 + np.exp(post[:, 2])

    # 后验中位数（可作为 plug-in 拟合量）
    mu_med = float(np.median(mu_samps))
    sig_med = float(np.median(sigma_samps))
    nu_med = float(np.median(nu_samps))

    # 快速拟合图（直方图+QQ）
    quick_fit_plots(r, mu_med, sig_med, nu_med,
                    out_prefix=os.path.join(OUT_DIR, "t_fit"))

    # ---- 生成 posterior 直方图（μ、σ、ν） ----
    plot_hist(mu_samps,    "Posterior of μ", "mu (%)",
              os.path.join(OUT_DIR, "post_mu_hist.png"))
    plot_hist(sigma_samps, "Posterior of σ", "sigma (%)",
              os.path.join(OUT_DIR, "post_sigma_hist.png"))
    plot_hist(nu_samps,    "Posterior of ν", "nu (df)",
              os.path.join(OUT_DIR, "post_nu_hist.png"))

    # ---- 生成 trace 图（μ、σ、ν） ----
    plot_trace(mu_samps,    "Trace: μ", "mu (%)",
               os.path.join(OUT_DIR, "trace_mu.png"))
    plot_trace(sigma_samps, "Trace: σ", "sigma (%)",
               os.path.join(OUT_DIR, "trace_sigma.png"))
    plot_trace(nu_samps,    "Trace: ν", "nu (df)",
               os.path.join(OUT_DIR, "trace_nu.png"))
    
    # ACF 图：分别对 μ、σ、ν 画
    plot_acf_series(mu_samps,    "μ chain", os.path.join(OUT_DIR, "acf_mu.png"),    max_lag=100)
    plot_acf_series(sigma_samps, "σ chain", os.path.join(OUT_DIR, "acf_sigma.png"), max_lag=100)
    plot_acf_series(nu_samps,    "ν chain", os.path.join(OUT_DIR, "acf_nu.png"),    max_lag=100)
    print("\n=== ESS (ACF-based, per chain) ===")
    print("ESS(mu)   :", int(ess_from_acf(mu_samps,    max_lag=200)))
    print("ESS(sigma):", int(ess_from_acf(sigma_samps, max_lag=200)))
    print("ESS(nu)   :", int(ess_from_acf(nu_samps,    max_lag=200)))


    # 汇总表
    summ = pd.DataFrame({
        "param":  ["mu(%)","sigma(%)","nu"],
        "median": [mu_med, sig_med, nu_med],
        "p2.5":   [np.percentile(mu_samps, 2.5),
                   np.percentile(sigma_samps, 2.5),
                   np.percentile(nu_samps, 2.5)],
        "p97.5":  [np.percentile(mu_samps, 97.5),
                   np.percentile(sigma_samps, 97.5),
                   np.percentile(nu_samps, 97.5)]
    })
    summ_path = os.path.join(OUT_DIR, "posterior_summary.csv")
    summ.to_csv(summ_path, index=False)

    # 控制台打印关键信息
    print("=== Data ===")
    print(f"CSV: {CSV_PATH}")
    print("\n=== Prior hyper-parameters (data-adaptive) ===")
    print(f"tau_mu = {tau_mu:.4f} (%),  c_sigma = {c_sigma:.4f} (%),  mean_nu_star = {mean_nu_star:.1f}")
    print("\n=== Acceptance rates (mu, log_sigma, eta) ===")
    print(acc_rate)
    print("\n=== Posterior summaries (median [2.5%, 97.5%]) ===")
    for name, arr in zip(["mu(%)","sigma(%)","nu"], [mu_samps, sigma_samps, nu_samps]):
        print(f"{name:8s}: {np.median(arr):.4f}  [{np.percentile(arr,2.5):.4f}, {np.percentile(arr,97.5):.4f}]")
    print("\nSaved to:", OUT_DIR)

if __name__ == "__main__":
    main()
