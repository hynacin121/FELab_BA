---

category: [FinancialDataAnalysis]
title : "Single Factor Models : EWMA Estimates"
excerpt : ""

date: 2022-02-17
use_math : true
---

# __EWMA Estimates__

## __Time-Varying Beta__

+ OLs won't reflect current market condition as it represents average value over the time period
+ The time varying beta estimates better reflect the current risk factor sensitivity


## __EWMA__
+ EWMA can be defined on any time series of data
+ Supopose that on date _t_ we have recorded data up to time _t-1_
+ then the EWMA of these observations is defined as

$
EWMA(x_{t-1}, \dots, x_1 \vert \; \lambda) = 
\frac{x_{t-1} + \lambda x_{t-2} + \lambda^2 x_{t-3} + \dots  + \lambda^{t-2} x_{1}}{1 + \lambda + \lambda^2 + \dots  + \lambda^{t-2}}
$

+ Also, since $ 1 + \lambda + \lambda^2 + \dots  = (1- \lambda)^{-1}$ . For large t,we can write $EWMA = (1- \lambda)\sum_{i=1}^\infty \lambda^{i-1}x_{t-i}$

+ We can use this formula to calculate EWMA estimates of 
    + variance, where we take x to be the squared returns
    + covariance, where we take x to be the cross product of the two returns

+ EWMA beta can be estimated as 
$
\hat{\beta}_{t\lambda} =\frac{Cov_\lambda(X_t,Y_t)}{V_\lambda(X_t)}
$

where $X_t$ denotes the market, and $ V_\lambda(r_t) = \hat{\sigma}_t^2$ and $Cov_\lambda(r_{1,t}, r_{2,t}) = \hat{\sigma}_{12t}$
