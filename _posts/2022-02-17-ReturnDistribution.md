---

category: [FinancialDataAnalysis]
title : "Return Distribution"
excerpt : "Stock Returns Distribution and compare with normal distribution"

date: 2022-02-17
use_math : true
---

# __Normally Distrubuted?__

+ Stock Returns have fatter tails compared to a normal distribution
+ We can also check the tailedness with a Q-Q plot

    Ex) S&P500 daily returns
<p align = "center">
<img src = "/assets/img/QQPlot.jpg" , height="300x", width="300px" >
</p>

# __Test for checking normality__

## __Kolmogorov-Smirnov test (K-S Test)__

$
 H_0 : Prob\; of\; Sample = P_0 \\
 H_A : Prob\; of\; Sample \neq P_0 \\
where\; P_0\; could \; be\; a\; normal\; distribution
$
  
+ Compare empirical cdf with the cdf of normal distribution
+ Empirical cdf :
$
F_n(x) = P_n(X \leq x) = \frac1n\sum_{i=1}^nI(X_i \leq x) \\
I(X_\leq x) = \begin{pmatrix} 1 \quad (True) \\ 0 \quad (false)\end{pmatrix}
$

Test statistics: 
$
\sup_{x \inc R}
$



## __Jarque-Bera test ( J-B Test)__

+ J-B Test is defined by
$
 JB = \frac n6 \cdot (S^2 + \frac{(K-3)^2}4)
$
where the sample skewness $ S = {\hat{\mu_3}}/{\hat{\mu_2}}^{3/2}$  is an estimator of $\beta_1 = \mu_3/\mu_2^{3/2} $ and sample kurtosis $ K = {\hat{\mu_4}}/{\hat{\mu_2}}^2 $ an estimator of $\beta_2 = \mu_4/\mu_2^2 $ amd the $\mu_2 \; and \; \mu_3$ are the thoerical second, third central moments, respectively, with its estimates
 $ 
 \hat{\mu_j} = \frac1n\sum_{i=1}^n(x_i - \bar{x})^j, \quad j=2,3,4
 $ 

# __Individual Stocks vs. Stock Markets__

+ Stock market returns display negative skewness.
+ Stock returns display positive skewness.

