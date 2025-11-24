import os
import pandas as pd
import numpy as np
from scipy import stats

# ===================================================================
# 1) 全局配置
# ===================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Tesla_close.csv")  # 数据文件路径
RETURNS_ARE_PERCENT = False        # CSV 中的收益率是否已经是百分比形式

OUT_DIR = os.path.join(BASE_DIR, "outputs")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

N_ITER    = 20000  # MCMC 总迭代次数
BURN_FRAC = 0.5    # Burn-in 比例

# MH 步长 (需要根据接受率进行调整)
STEP_MU   = 0.04
STEP_LOGS = 0.05
STEP_ETA  = 0.05

SEED = 42 # 随机种子，保证结果可复现

# ===================================================================

# ============ 工具函数 ============

def mad_robust(x):
    """稳健尺度：1.4826 * MAD"""
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def load_returns(csv_path, returns_are_percent=True):
    """从 CSV 加载收益率数据。"""
    df = pd.read_csv(csv_path)
    # 假设收益率数据在名为 'Log_Return' 的列中
    if 'Log_Return' not in df.columns:
        raise ValueError("Column 'Log_Return' not found in CSV file.")
    r = df['Log_Return'].dropna().values
    if not returns_are_percent:
        r = r * 100.0
    return r

# ---------- 先验 ----------
def logprior(theta, tau_mu, c_sigma, mean_nu_star):
    """
    先验的对数：
      mu ~ N(0, tau_mu^2)
      sigma ~ Half-Cauchy(0, c_sigma)    （在真实尺度 sigma 上）
      (nu-2) ~ Exponential(mean = mean_nu_star)
    注意：代码在 (log_sigma, eta) 空间工作，需加 Jacobian：+log_sigma + eta
    """
    mu, log_sigma, eta = theta
    sigma   = np.exp(log_sigma)      # sigma > 0
    nu_star = np.exp(eta)            # nu - 2 > 0

    # mu 的正态先验
    lp = stats.norm.logpdf(mu, 0.0, tau_mu)

    # sigma 的 Half-Cauchy 先验（对数密度）
    # f(sigma) = [2 / (pi * c)] * 1 / (1 + (sigma/c)^2), sigma>0
    lp += np.log(2.0) - np.log(np.pi * c_sigma) - np.log(1.0 + (sigma / c_sigma)**2)

    # (nu-2) 的指数先验（均值 mean_nu_star；rate = 1/mean）
    rate = 1.0 / mean_nu_star
    lp += np.log(rate) - rate * nu_star

    # 变量变换的 Jacobian：sigma = exp(log_sigma) → +log_sigma；nu-2 = exp(eta) → +eta
    lp += log_sigma + eta
    return lp

# ---------- 层级表示下的似然 ----------
def loglik_normal_given_lambda(theta, r, lam):
    """
    给定隐变量 λ_t 的条件似然（对数）：
      r_t | λ_t ~ Normal( mu,  sigma^2 / λ_t )
    这里 theta = (mu, log_sigma, eta)，eta 只影响抽 λ 的条件式，这个似然里不直接用到 eta。
    """
    mu, log_sigma, _ = theta
    sigma = np.exp(log_sigma)  # 真实尺度
    # 对每个 t：方差 = sigma^2 / lam_t
    var_t = (sigma ** 2) / lam
    # 正态对数密度逐点求和（数值稳定地拆分）
    # log N(r | mu, var_t) = -0.5*log(2π) - 0.5*log(var_t) - (r-mu)^2 / (2*var_t)
    ll = -0.5 * np.log(2.0 * np.pi) * len(r)
    ll += -0.5 * np.sum(np.log(var_t))
    ll += -0.5 * np.sum(((r - mu) ** 2) / var_t)
    return ll

def logpost_hier(theta, r, lam, tau_mu, c_sigma, mean_nu_star):
    """
    层级模型下的对数后验：
      log posterior = loglik_normal_given_lambda + logprior
    注意：这里的似然使用了当前 λ 序列（在每轮迭代最先用 Gibbs 更新得到）
    """
    return loglik_normal_given_lambda(theta, r, lam) + logprior(theta, tau_mu, c_sigma, mean_nu_star)

# ---------- 关键：Gibbs 抽样 λ_t ----------
def gibbs_sample_lambda(r, mu, sigma, nu):
    """
    对整列 λ_{1:n} 做一次 Gibbs 更新：
      λ_t | r_t, mu, sigma, nu ~ Gamma( (nu+1)/2,  rate = (nu + ((r_t-mu)^2 / sigma^2)) / 2 )
    SciPy: gamma(shape=k, scale=θ) —— 其中 θ = 1/rate
    返回：lam (shape = [n,])
    """
    n = len(r)
    shape = (nu + 1.0) / 2.0
    # rate_t = ( nu + ((r_t - mu)^2 / sigma^2) ) / 2
    rate_t = (nu + ((r - mu) ** 2) / (sigma ** 2)) / 2.0
    scale_t = 1.0 / rate_t  # SciPy 需要的是 scale = 1/rate

    # 对每个 t 独立采样（向量化）
    # 注意：scipy.stats.gamma 支持向量化的 scale
    lam = stats.gamma.rvs(a=shape, scale=scale_t, size=n)
    return lam

# ============ Metropolis-within-Gibbs 采样器（最小改动版） ============

def mwg_sampler(r,
                n_iter=15000, burn_frac=0.5,
                step_mu=0.05, step_logsig=0.05, step_eta=0.05,
                tau_mu=0.5, c_sigma=1.5, mean_nu_star=30.0,
                init=None, random_seed=42):
    """
    Metropolis-within-Gibbs（MwG）：
      1) Gibbs：给定 (mu, sigma, nu) 对整列 λ 采样（可解析，必收）
      2) MH：依次对 (mu, log_sigma, eta) 做随机游走提议并接受/拒绝
    与你原来的 MH 采样器的唯一区别：每轮迭代开始，先更新 λ
    """
    rng = np.random.default_rng(random_seed)

    # ------ 初始化 θ 和 λ ------
    if init is None:
        s_robust = mad_robust(r)
        mu0   = 0.0
        logs0 = np.log(max(s_robust, 1e-8))
        eta0  = np.log(8.0 - 2.0)   # 初始 nu ≈ 8
        theta = np.array([mu0, logs0, eta0], dtype=float)
    else:
        theta = np.array(init, dtype=float)

    # 初始 λ：用当前参数的条件式 Gibbs 一次（或全 1 也可）
    mu, log_sigma, eta = theta
    sigma = np.exp(log_sigma)
    nu    = 2.0 + np.exp(eta)
    lam   = gibbs_sample_lambda(r, mu, sigma, nu)

    steps   = np.array([step_mu, step_logsig, step_eta], dtype=float)
    acc     = np.zeros(3, dtype=int)
    chain   = np.zeros((n_iter, 3), dtype=float)  # 保存 θ 的轨迹
    lp_cur  = logpost_hier(theta, r, lam, tau_mu, c_sigma, mean_nu_star)

    # ------ 主循环 ------
    for it in range(n_iter):
        # (A) 先对整列 λ 做一次 Gibbs 更新（必收）
        mu, log_sigma, eta = theta
        sigma = np.exp(log_sigma)
        nu    = 2.0 + np.exp(eta)
        lam   = gibbs_sample_lambda(r, mu, sigma, nu)

        # (B) 再对 θ 的三个坐标做 MH 更新（与原来相同，只是似然换成用了 lam 的 normal 似然）
        for j in range(3):
            prop = theta.copy()
            prop[j] += rng.normal(0.0, steps[j])  # 对称正态提议

            # 计算候选点的对数后验（注意：似然使用给定 lam 的 normal 形式）
            lp_prop = logpost_hier(prop, r, lam, tau_mu, c_sigma, mean_nu_star)

            # MH 接受判据：log(u) <= logpost_prop - logpost_cur
            if np.log(rng.random()) <= (lp_prop - lp_cur):
                theta  = prop
                lp_cur = lp_prop
                acc[j] += 1

        chain[it, :] = theta

    # ------ 返回后验样本与接受率 ------
    burn     = int(burn_frac * n_iter)
    post     = chain[burn:, :]
    acc_rate = acc / n_iter
    return post, acc_rate, chain


# --------------------------
# 5) 便捷图表函数 (美化版)
# --------------------------
import matplotlib.pyplot as plt

def plot_hist(x, title, xlabel, outpath):
    """Beautified histogram with an overlaid KDE curve."""
    with plt.style.context('seaborn-v0_8-darkgrid'):
        fig = plt.figure(figsize=(10, 6))
        ax  = fig.add_subplot(111)
        # 绘制直方图
        ax.hist(x, bins=50, density=True, alpha=0.7, label="Posterior samples histogram")
        # 叠加 KDE 曲线
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
        # 绘制均值线
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
        # 1. 直方图 + 拟合 t 密度
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
        # 使用 stats.probplot 可以更方便地生成 QQ 图
        stats.probplot(r, dist=stats.t, sparams=(nu_hat, mu_hat, sig_hat), plot=ax2)
        # 美化 probplot 的输出
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
    计算样本自相关函数 ρ(k)，k=0..max_lag。
    ρ(0)=1。实现基于相关/方差的标准化。
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
        # 绘制置信区间 (95%, 1.96/sqrt(n))
        conf_level = 1.96 / np.sqrt(n)
        ax.fill_between(lags, -conf_level, conf_level, color='lightblue', alpha=0.5, label="95% confidence band")
        # 绘制 ACF 棒棒糖图
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
    """主流程：加载数据 -> 运行采样器 -> 分析和绘图"""
    # 读收益（单位：百分比）
    r = load_returns(CSV_PATH, returns_are_percent=RETURNS_ARE_PERCENT)

    # (动态设定先验超参数)
    s_robust = mad_robust(r)
    hyper_params = {
        "tau_mu":       s_robust,      # mu 的先验标准差，与数据尺度相关
        "c_sigma":      s_robust,      # sigma 的先验尺度
        "mean_nu_star": 30.0           # nu-2 的先验均值（无信息）
    }

    # 运行采样器
    post, acc_rate, chain = mwg_sampler(r,
                                        n_iter=N_ITER,
                                        burn_frac=BURN_FRAC,
                                        step_mu=STEP_MU,
                                        step_logsig=STEP_LOGS,
                                        step_eta=STEP_ETA,
                                        random_seed=SEED,
                                        **hyper_params)

    # 分析并出图
    analyze_and_plot_results(r=r,
                             post=post,
                             chain=chain,
                             acc_rate=acc_rate,
                             sampler_name="Metropolis-within-Gibbs",
                             hyper_params=hyper_params)

def analyze_and_plot_results(r, post, chain, acc_rate, sampler_name, hyper_params):
    """
    对 MCMC 采样结果进行后处理、分析、绘图和保存。
    """
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

    # ---- 生成后验分布直方图（μ、σ、ν） ----
    plot_hist(mu_samps,    "Posterior of μ", "mu (%)",
              os.path.join(OUT_DIR, "post_mu_hist.png"))
    plot_hist(sigma_samps, "Posterior of σ", "sigma (%)",
              os.path.join(OUT_DIR, "post_sigma_hist.png"))
    plot_hist(nu_samps,    "Posterior of ν", "nu (df)",
              os.path.join(OUT_DIR, "post_nu_hist.png"))

    # ---- 生成轨迹图（μ、σ、ν），使用完整 chain ----
    plot_trace(chain[:, 0], "Trace: μ", "mu (%)",
               os.path.join(OUT_DIR, "trace_mu.png"))
    plot_trace(np.exp(chain[:, 1]), "Trace: σ", "sigma (%)",
               os.path.join(OUT_DIR, "trace_sigma.png"))
    plot_trace(2.0 + np.exp(chain[:, 2]), "Trace: ν", "nu (df)",
               os.path.join(OUT_DIR, "trace_nu.png"))
    
    # ---- ACF 图：分别对 μ、σ、ν 画 ----
    plot_acf_series(mu_samps,    "μ chain", os.path.join(OUT_DIR, "acf_mu.png"),    max_lag=100)
    plot_acf_series(sigma_samps, "σ chain", os.path.join(OUT_DIR, "acf_sigma.png"), max_lag=100)
    plot_acf_series(nu_samps,    "ν chain", os.path.join(OUT_DIR, "acf_nu.png"),    max_lag=100)

    # 汇总表
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

    summary_path = os.path.join(OUT_DIR, "posterior_summary.csv")
    summary.to_csv(summary_path, index=False)

    acc_df = pd.DataFrame({
        "parameter": ["mu", "log_sigma", "eta"],
        "accept_rate": acc_rate,
    })
    acc_df.to_csv(os.path.join(OUT_DIR, "acceptance_rates.csv"), index=False)

    diagnostics_txt = os.path.join(OUT_DIR, "diagnostics.txt")
    with open(diagnostics_txt, "w", encoding="utf-8") as fh:
        fh.write(f"Sampler: {sampler_name}\n")
        fh.write(f"Iterations: {len(chain)}, burn-in fraction: {BURN_FRAC}\n")
        fh.write("Acceptance rates (mu, log_sigma, eta): " + ", ".join(f"{x:.3f}" for x in acc_rate) + "\n")
        fh.write("ESS (mu, sigma, nu): " + ", ".join(f"{x:.1f}" for x in ess_vals) + "\n")
        fh.write(f"Summary CSV: {summary_path}\n")

    print("Posterior summaries and diagnostics saved to the outputs/ folder.")


if __name__ == "__main__":
    main()