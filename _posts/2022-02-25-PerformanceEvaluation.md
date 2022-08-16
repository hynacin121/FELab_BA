---
category: [FinancialDataAnalysis]
title : "Performance Evaluation"
excerpt : ""

date: 2022-02-25
use_math : true
---

# __Performance Measures__

+ Return, Risk
+ Risk-adjusted return : Sharpe ratio
+ Market risk : Beta
+ Downside risk : MDD, VaR, CVaR

# __Annualized Return(Geometric mean)__

+ With the assumption that monthly return of the portfolio will consistently by $r_H$ during the next 12 months, the annual return $r_A$ becomes
$
r_A = (1+r_{0,1})(1+r_{1,2})\dots(1+r_{11,12})-1 = (1+r_H)^{12} -1
$

# __Sharpe Ratio__
+ Computes the expected excess return of portfolio for every unit of volatility it takes
$
Sharpe\; ratio = \frac{E(r)-r_f}{\sigma(r)}
$
+ One of the most widely used measures of performance
+ Be careful with negative Sharpe ratios
    + Smaller Sharpe Ratio is better, if they are negative.
+ Sharpe Ratio is not time-horizon independent

# __Alpha and Beta__

+ Beta is called non-diversifiable risk because it is the portion of risk that cannot be diversified away based on the MPT
+ Beta is from the CAPM
$
E(r) = r_f + \beta(E(r_m)- r_f)\\
\beta = \frac{cov(r,r_m)}{Var(r_m)}
$

# __Tracking Error__

+ Relative performance is of greater interest, such as the excess return over the risk-free rate.
+ Most portfolio managers regularly assess their performance relative to a benchmark
+ Tracking error shows how closely a portfolio follows a given bench mark
$
Tracking \; Error = \sqrt{Var(r_i - r_{b,i})}
$
where $r_b \in R^T$ is the benchmark return
+ Tracking error is the standard deviation of active return.
$
Active \; Return = Portfolio \; Return - Benchmark \; Return
$

+ Similar to the Volatility of Portfolio Return, an investor seeking a portfolio with low risk will prefer a low tracking error in general.
+ Thus a passive incestor will cetainly reduce tracking error when performance is evaluated relative to benchmark

# __Information Ratio__

+ Information ratio is evaluated by dividing the excess portfolio return over the benchmark by the tracking error,
$
Information\; ratio = \frac{E(r-r_b)}{\sigma(r-r_b)} 
$ where $r_b$ is the benchmark return

+ Info ratio is higher for portfolios that closely follow the benchmark but have higher expected returns

# __Sortino Ratio__

+ Sortino Ratio uses downside deviation as a measure of risk
$
Downside \;deviation = \sqrt{\frac1{T-1}\sum_{i=1}^T(min(0,r_i-r_{MAR}))^2
}
$
where $r_{MAR}$ is the minimum acceptable return or the desired target return

$
Sortino \; ratio = \frac{E(r)-r_{MAR}}{(Downside \; deviation)_{MAR=r_{MAR}}}
$

# __Maximum Drawdown__

+ Drawdown is computed for a specific time horizon. Drawdown at time T measures a decline from its previous peak during its investment horizon:
$
DD_T = max(0, \max_{t \in(0,T)}(\frac{P_t-P_T}{P_t}))
$

<p>
<img src = "/assets/img/MDD.png"  >
</p>
    

$
MDD_T =  \max_{t \in (0,T)}DD_T =  max(0, \max_{t \in(0,T)}(\frac{P_t-P_T}{P_t}))
$

# __Value at Risk__

+ Similar to maximum drawdown, VaR also attempts to find the worstcase loss of an invenstment

+ VaR is a probabilistic approach VaR at 95% represents the minimum level of loss that is only exceeded with, at most, 5% probability

$
VaR_\alpha = min[l\; \vert \; P(l_p > l) \leq 1 -\alpha]  
$

## __Limitation of Var__

+ VaR can be difficult to estimate, and different estimation methods can give  quite different value
+ VaR for individual positions doesn't aggregate in a simply way to portfolio VaR
+ VaR doesn't not fully represent returns beyond the minimum loss

# __Conditional Value_at_Risk__

+ CVaR overcomes this drawback by computing the expected return of scenarios that are worse than VaR level
$
CVaR_\alpha = E(l \; \vert \; l \geq VaR_\alpha)
\\ = \frac1{1-\alpha}\int_{-r'\omega}(-r'\omega)p(r)dr
$
where $-r'\omega$ is return and $p(r)$ is probability
