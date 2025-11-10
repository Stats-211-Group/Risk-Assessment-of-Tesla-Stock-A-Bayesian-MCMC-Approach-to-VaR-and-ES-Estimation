# -*- coding: utf-8 -*-
"""
Stat 211 Project - Student-t model for daily returns (Tesla example)
- Reads CSV with columns: Date, Close, Log_Return
- Uses a data-adaptive weakly-informative prior
- Estimates posterior of (mu, sigma, nu) via random-walk Metropolis-Hastings
- Produces: posterior summaries, VaR/ES (plug-in), hist & QQ, and posterior hist/trace for mu/sigma/nu

Author: your team
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------
# 0) 路径与配置（按你的机器改这里）
# --------------------------
CSV_PATH = r"D:\A_Python_Project\Stat 211 Project\Tesla_close.csv"
OUT_DIR  = r"D:\A_Python_Project\Stat 211 Project\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# 如果 Log_Return 列已经是“百分比”（例如 0.35 表示 0.35%），设为 True。
# 如果 Log_Return 是“小数收益”（例如 0.0035 表示 0.35%），设为 False，脚本会 *100 变成百分比。
RETURNS_ARE_PERCENT = False

# 采样设置（可根据接受率微调步长）
N_ITER     = 15000      # 总迭代
BURN_FRAC  = 0.50       # 丢弃前 50%
STEP_MU    = 0.05       # mu 的随机游走步长（单位：%，因为我们用的是百分比）
STEP_LOGS  = 0.05       # log(sigma) 的步长
STEP_ETA   = 0.05       # eta = log(nu-2) 的步长
SEED       = 42

# --------------------------
# 1) 数据加载与预处理
# --------------------------
def load_returns(csv_path, returns_are_percent=False):
    """读取 CSV，如果有 Log_Return 列就直接用；否则用 Close 计算。
    返回：一维 numpy 数组 r（单位：百分比 %）
    """
    df = pd.read_csv(csv_path)
    if "Log_Return" in df.columns:
        r = df["Log_Return"].astype(float).values
        # 若是小数收益，则 *100 变百分比
        if not returns_are_percent:
            r = r * 100.0
    else:
        # 没有 Log_Return 就用 Close 计算对数收益
        px = df["Close"].astype(float).values
        r = np.diff(np.log(px))
        r = r * 100.0  # 转成百分比
    return r

def mad_robust(x):
    """稳健尺度：s_robust = 1.4826 * MAD"""
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

# --------------------------
# 2) 模型：log-likelihood（t 分布，位置-尺度参数化）
#    参数工作变量：theta = (mu, log_sigma, eta)，其中
#       sigma = exp(log_sigma) > 0
#       nu    = 2 + exp(eta)   > 2  （保证方差存在）
# --------------------------
def loglik_t(theta, r):
    mu, log_sigma, eta = theta
    sigma = np.exp(log_sigma)
    nu    = 2.0 + np.exp(eta)
    # 对每个 r_t 取 t 分布的 logpdf 并求和
    return np.sum(stats.t.logpdf(r, df=nu, loc=mu, scale=sigma))

# --------------------------
# 3) 先验：弱信息、与数据尺度对齐
#    mu ~ N(0, tau_mu^2)
#    sigma ~ Half-Cauchy(0, c_sigma)
#    nu-2 ~ Exponential(mean = mean_nu_star)
#    注意：因为代码在 (log_sigma, eta) 上工作，需要加入 Jacobian：+log_sigma + eta
# --------------------------
def logprior(theta, tau_mu, c_sigma, mean_nu_star):
    mu, log_sigma, eta = theta
    sigma   = np.exp(log_sigma)
    nu_star = np.exp(eta)  # = nu - 2

    # mu 的正态先验
    lp = stats.norm.logpdf(mu, 0.0, tau_mu)

    # sigma 的 Half-Cauchy 先验（对数形式）
    # f(sigma) = [2 / (pi * c)] * 1 / (1 + (sigma/c)^2), sigma>0
    lp += np.log(2.0) - np.log(np.pi * c_sigma) - np.log(1.0 + (sigma / c_sigma)**2)

    # (nu-2) 的指数先验（均值 mean_nu_star，对应 rate=1/mean）
    rate = 1.0 / mean_nu_star
    lp += np.log(rate) - rate * nu_star

    # 变量变换的 Jacobian：sigma = exp(log_sigma) => +log_sigma；nu-2 = exp(eta) => +eta
    lp += log_sigma + eta
    return lp

# 后验 = 似然 + 先验（差一个常数，采样时不需要）
def logpost(theta, r, tau_mu, c_sigma, mean_nu_star):
    return loglik_t(theta, r) + logprior(theta, tau_mu, c_sigma, mean_nu_star)

# --------------------------
# 4) 随机游走 Metropolis-Hastings 采样器
#    依次更新 (mu, log_sigma, eta)，每个维度做一个对称正态提议
# --------------------------
def mh_sampler(r,
               n_iter=15000, burn_frac=0.5,
               step_mu=0.05, step_logsig=0.05, step_eta=0.05,
               tau_mu=0.5, c_sigma=1.5, mean_nu_star=30.0,
               init=None, random_seed=42):
    rng = np.random.default_rng(random_seed)

    # 初始化：mu≈0，sigma≈s_robust，nu≈8~12
    if init is None:
        s_robust = mad_robust(r)
        mu0   = 0.0
        logs0 = np.log(max(s_robust, 1e-8))
        eta0  = np.log(8.0 - 2.0)   # nu0 ≈ 8
        theta = np.array([mu0, logs0, eta0], dtype=float)
    else:
        theta = np.array(init, dtype=float)

    steps   = np.array([step_mu, step_logsig, step_eta], dtype=float)
    acc     = np.zeros(3, dtype=int)
    chain   = np.zeros((n_iter, 3), dtype=float)
    lp_cur  = logpost(theta, r, tau_mu, c_sigma, mean_nu_star)

    for it in range(n_iter):
        for j in range(3):
            prop      = theta.copy()
            prop[j]  += rng.normal(0.0, steps[j])  # 对称提议：N(theta_j, step_j^2)
            lp_prop   = logpost(prop, r, tau_mu, c_sigma, mean_nu_star)
            # 接受概率 alpha = min(1, exp(lp_prop - lp_cur))
            if np.log(rng.random()) <= (lp_prop - lp_cur):
                theta  = prop
                lp_cur = lp_prop
                acc[j] += 1
        chain[it, :] = theta

    burn     = int(burn_frac * n_iter)
    post     = chain[burn:, :]
    acc_rate = acc / n_iter
    return post, acc_rate, chain

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
