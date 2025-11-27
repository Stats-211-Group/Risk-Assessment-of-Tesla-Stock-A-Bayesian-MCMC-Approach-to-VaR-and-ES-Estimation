# Risk Assessment of Tesla Stock: A Bayesian MCMC Approach to VaR and ES Estimation

**Course:** STATS 211
**Team Members:** Ke Ning; Yanpei Yu; Ziying Ye
**Date:** Nov 27th, 2025

---

## 1. Introduction

This project aims to model the daily returns of Tesla Inc. (TSLA) using Bayesian statistical methods and to quantify its tail risk. Financial asset returns often exhibit "leptokurtic and fat-tailed" characteristics, meaning extreme market movements occur more frequently than predicted by a standard Gaussian distribution. Consequently, traditional risk models based on normality assumptions often underestimate potential losses.

To address this, we employ a **Student's t-distribution** model, implemented via a **Metropolis-within-Gibbs** sampling algorithm. This allows us to estimate the posterior distributions of the model parameters ($\mu, \sigma, \nu$). Based on these posterior samples, we calculate the **Value at Risk (VaR)** and **Expected Shortfall (ES)** at a 95% confidence level, providing a robust quantitative basis for risk management.

## 2. Data Description

* **Source:** Yahoo Finance / Public Market Data
* **Time Period:** January 4, 2022 – December 31, 2024 (3 Years)
* **Preprocessing:**
  Daily closing prices ($P_t$) were converted into log-returns ($r_t$) and scaled to percentages for numerical stability:
  $$
  t = 100 \times \ln\left( \frac{P_t}{P_{t-1}} \right)

  $$
* **Observation:** The dataset covers a period of significant volatility for Tesla, including multiple sharp drawdowns, confirming the necessity of a fat-tailed model.

## 3. Methodology

### 3.1 Model Specification: Scale Mixture of Normals

Directly sampling from a t-distribution likelihood can be computationally challenging. We utilize the **scale mixture of normals** representation, which introduces a latent variable $\lambda_t$ for each observation. This transforms the model into a hierarchical structure that is easier to sample using Gibbs steps:

$$
\begin{aligned}
r_t | \mu, \sigma^2, \lambda_t &\sim \mathcal{N}\left(\mu, \frac{\sigma^2}{\lambda_t}\right) \\
\lambda_t | \nu &\sim \text{Gamma}\left(\frac{\nu}{2}, \frac{\nu}{2}\right)
\end{aligned}

$$

Where:

* $\mu$: Location parameter (mean daily return).
* $\sigma$: Scale parameter (base volatility).
* $\nu$: Degrees of freedom, controlling tail thickness ($\nu \to \infty$ implies normality).
* $\lambda_t$: Latent weight for day $t$. A small $\lambda_t$ indicates a high-variance (outlier) observation.

### 3.2 Priors

We adopt the following prior distributions for the parameters $\theta = (\mu, \sigma, \nu)$:

* **$\mu \sim \mathcal{N}(0, \tau_\mu^2)$**: Weakly informative normal prior.
* **$\sigma \sim \text{Half-Cauchy}(0, C_\sigma)$**: Prior on the scale parameter.
* **$\nu - 2 \sim \text{Exponential}(\beta)$**: Since $\nu > 2$ is required for finite variance, we model the shifted parameter.

### 3.3 The Metropolis-within-Gibbs Sampler

We implemented a hybrid MCMC sampler that alternates between exact conditional sampling (Gibbs) and approximate updates (Metropolis-Hastings).

**Step 1: Gibbs Update for Latent Variables ($\lambda_t$)**
Conditional on the data $r_t$ and current parameters $(\mu, \sigma, \nu)$, the posterior for each weight $\lambda_t$ follows a Gamma distribution:

$$
\lambda_t | r_t, \mu, \sigma, \nu \sim \text{Gamma}\left( \frac{\nu + 1}{2}, \frac{\nu + \left(\frac{r_t - \mu}{\sigma}\right)^2}{2} \right)

$$

We sample the vector $\boldsymbol{\lambda} = (\lambda_1, \dots, \lambda_n)$ directly in this step.

**Step 2: Metropolis-Hastings Update for Parameters ($\theta$)**
The parameters $\theta = (\mu, \ln\sigma, \ln(\nu-2))$ do not have closed-form conditional posteriors. We update them using a Random Walk Metropolis step.
For a proposed parameter set $\theta^*$:

1. **Proposal:** Generate $\theta^* \sim \mathcal{N}(\theta^{(t-1)}, \Sigma_{\text{step}})$.
2. **Acceptance Probability ($\alpha$):**

   $$
   \alpha = \min\left(1, \frac{P(\mathbf{r} | \boldsymbol{\lambda}, \theta^*) P(\boldsymbol{\lambda} | \nu^*) P(\theta^*)}{P(\mathbf{r} | \boldsymbol{\lambda}, \theta^{(t-1)}) P(\boldsymbol{\lambda} | \nu^{(t-1)}) P(\theta^{(t-1)})} \right)

   $$

   *Note: It is crucial to include the $P(\boldsymbol{\lambda}|\nu)$ term in the posterior calculation, as $\nu$ influences the likelihood of the latent variables.*
3. **Decision:** Draw $u \sim \text{Uniform}(0,1)$. If $u < \alpha$, set $\theta^{(t)} = \theta^*$; otherwise, keep $\theta^{(t)} = \theta^{(t-1)}$.

## 4. Diagnostics & Validation

To ensure the validity of our MCMC results, we performed standard convergence checks.

### 4.1 Trace Plots

* **Analysis:** The trace plots for $\mu$, $\sigma$, and $\nu$ exhibit a stable "caterpillar" pattern with no discernible trend. This indicates that the chain has successfully converged to the stationary distribution after the burn-in period.

<div align="center">
    <img src="outputs\trace_mu.png" width="48%"> 
    <img src="outputs\trace_sigma.png" width="48%">
</div>

<br> <div align="center">
    <img src="outputs\trace_nu.png" width="50%">
</div>

### 4.2 Autocorrelation (ACF)

* **Analysis:** High autocorrelation is expected in this hierarchical model due to the strong coupling between $\sigma$ and $\boldsymbol{\lambda}$. We addressed this by running a sufficiently long chain (20,000+ iterations) to ensure an adequate Effective Sample Size (ESS).

<div align="center">
    <img src="outputs\acf_mu.png" width="48%"> 
    <img src="outputs\acf_sigma.png" width="48%">
</div>

<br> <div align="center">
    <img src="outputs\acf_nu.png" width="50%">
</div>

### 4.3 Goodness-of-Fit

* **Histogram:** The fitted t-distribution (red line) aligns well with the posterior data, capturing the central peak more effectively than a normal distribution.
* **QQ Plot:** The quantiles of the data closely follow the theoretical t-quantiles. Crucially, the extreme tails do not deviate significantly from the diagonal, validating the fat-tailed assumption.

<img src="outputs\t_fit_hist.png" width="50%">
<img src="outputs\t_fit_qq.png" width="40%">

## 5. Results & Risk Analysis

### 5.1 Posterior Estimates


| Parameter            | Mean Estimate | Interpretation                                                      |
| :------------------- | :------------ | :------------------------------------------------------------------ |
| **$\mu$ (Mu)**       | ~0.05%        | Average daily return is close to zero, typical for daily data.      |
| **$\sigma$ (Sigma)** | ~3.10%        | Base daily volatility is ~3.1%, indicating a high-volatility asset. |
| **$\nu$ (Nu)**       | **~7.35**     | Degrees of freedom$\nu \ll 30$ confirms significant **fat tails**.  |

### 5.2 Risk Measures (VaR & ES)

We calculated risk metrics at the 95% confidence level.

<img src="outputs\risk_posterior.png">


| Metric        | Estimate (Mean) | 95% Credible Interval | Business Insight                                                                                                     |
| :------------ | :-------------- | :-------------------- | :------------------------------------------------------------------------------------------------------------------- |
| **VaR (95%)** | **-6.09%**      | [-6.58%, -5.66%]      | In 95% of trading days, losses will not exceed**6.1%**. This serves as a "safety baseline" for margin requirements.  |
| **ES (95%)**  | **-8.69%**      | [-9.79%, -7.88%]      | If the VaR threshold is breached, the expected average loss is**8.7%**. This highlights the severity of tail events. |

### 5.3 Insight: Upside Potential

While VaR focuses on downside risk, the symmetric nature of the t-distribution ($\nu \approx 7.35$) implies equivalent upside potential. Our model estimates a **95% Upside Potential of +6.1%**, characterizing Tesla as a "High Risk, High Reward" asset with symmetric volatility.

## 6. Conclusion

This project successfully implemented a Bayesian t-distribution model to analyze Tesla's stock risk.

1. **Model Suitability:** The low estimated degrees of freedom ($\nu \approx 7.35$) strongly reject the normality assumption, justifying the use of the t-distribution.
2. **Risk Quantification:** The estimated ES of -8.69% reveals a significant gap from the VaR of -6.09%, warning of substantial losses during extreme market downturns.
3. **Implications:** Risk managers holding TSLA should capitalize for tail events significantly larger than standard deviation-based models would suggest.

---

*Appendix: Python code and diagnostic files are attached.*

```python

```
