---
category: [FinancialDataAnalysis]
title : "Portfolio Optimization"
excerpt : ""

date: 2022-02-19
use_math : true
---

# __Portfolio Optimization__

+ Decide how to divide an investor's wealth among n assets


# __Mean-Variance Optimization__

+ Markowitz models the trade-off between risk and returns as an optimization problem.
+ Efficient portfolio can be formed through diversification
+ Became the basis of Modern Portfolio Theory (MPT)

## __Impact of Markowitz__

+ Concepts of portfolio risk management and diversification have been instrumental in the development of modern financial decision making.
+ Investor should decide based on both the expected return and risk from their investment.

## __Diversification__

+ The definition of risk as the variance of returns leads to the conclusion that diversification is preferable as an investment strategy.


# __MVO Formulation__

+ Markowitz mean-variance portfolio formulation

$
\min_w (Portfolio Variance) \\
s.t. (Porfolio mean return) \geq \mu_0
$



## __MVO Inputs__
+ Required Inputs
    + Mean returns
    + Covariance of asset returns 

## __Portfolio Mean and Variance
+ Portfolio weight :  $\omega =  \begin{pmatrix}\omega_1 \\ \vdots \\ \omega_n\end{pmatrix}$

+ Portfolio return : $ r_{pfo} =\sum_i r_i\omega_i $
+ Portfolio mean return :
$ E[r_{pfo}] = \sum_iE[r_i]\omega_i = \sum_i\mu_i\omega_i = \mu^T\omega$
+ Portfolio variance :
$ Var(r_{pfo}) =\sum_i\sum_j\omega_i\omega_j\sigma_{ij} = \omega^T\Sigma\omega$

## __MVO Formulation__

+ Markowitz mean-variance portfolio formulation(1)

$
min\sum_{ij}\omega_i\omega_j\sigma_{ij}\quad \dots risk\\
s.t. \; \sum_i \omega_i\mu_i \geq \mu_i  \quad \dots return\\  
\sum_i\omega_i = 1
$

+ Markowitz mean-variance portfolio formulation(2)

$
\min_{\omega \in \Omega} \omega^T\Sigma\omega - \lambda\mu^T\omega
$
        $\lambda$ : risk-seeking coefficient \
        $\Omega$ : set of feasible portfolios

## __GMV Portfolio__

$
\min_\omega \omega^T\Sigma\omega\\
s.t.\; \omega^T\iota = 1
$

## __Additional Constraints__

+ Short selling restriction
 $
\min_\omega \omega^T\Sigma\omega\\
s.t.\; \mu^T\omega \geq \mu_0\\
\omega^T\iota = 1\\
\omega_i \geq 0
$

+ Allocation restriction
 $
\min_\omega \omega^T\Sigma\omega\\
s.t.\; \mu^T\omega \geq \mu_0\\
\omega^T\iota = 1\\
\omega_l \leq \omega_i \leq \omega_h
$


+ MVO with Turnover
    + Turnover : Percentage of portfolio that is sold in a particular month or year
$
\min_\omega \omega^T\Sigma\omega\\
s.t.\; \mu^T\omega \geq \mu_0\\
\omega^T\iota = 1\\
\left\vert \omega_i - \omega_{0,i}\right\vert \leq u_i
$


+ MVO with Transaction costs

$
TC = \sum_ic_i\left\vert \omega_i - \omega_{0,i}\right\vert 
$

+ MVO with Tracking Error
year
$
\min_\omega \omega^T\Sigma\omega\\
s.t.\; \mu^T\omega \geq \mu_0\\
\omega^T\iota = 1\\
(\omega - \omega_b)^T\Sigma(\omega- \omega_b) \leq \sigma_{TE}^2
$

+ MVO with Cardinality (Lasso)
    + Example of Lasso : 
    $
    \min_{\beta \in ℝ} {\frac1N \left\vert \right\vert y-X\beta \left\vert \right\vert_2^2 + \lambda \left\vert \right\vert \beta \left\vert \right\vert_1 }
    $ 

    + MVO with regularization

$
\min_\omega \; \omega^T\Sigma\omega + \lambda_1 \left\vert \right\vert \omega \left\vert \right\vert_1  + \lambda_2 \left\vert \right\vert \omega \left\vert \right\vert_2^2\\
s.t.\; \mu^T\omega \geq \mu_0\\
\omega^T\iota = 1\\
$

# __Additional Risk Measures__

## __Why use Variance as Risk?__

+ Variance penalizes both upside and downside deviation. Using the variance in the portfolio optimization context means that outcomes that are above the expected portfolio returned are deemed as risky as outcomes that are below the expected portfolio return.

## __Risk__

+ Divided into two classes : dispersion and downside risk measure

### __Dispersion__
+ Variance : $\sigma_p^2 = \omega'\Sigma \omega$ 
+ Absolute Varinace : $AD(\tilde{r}_p) = E[\left\vert \sum_{i=1}^N\omega_i \tilde{r}_i - \sum_{i=1}^N\omega_i \mu_i  \right\vert] = E[\left\vert\sum_ir_i\omega_i - E[\sum_ir_i\omega_i] \right\vert] $ 

### __Downside Risk Measure__
+ LSAD : $LSAD(r_{pfo}) = AD_(r_{pfo}) = E[\left\vert r_{pfo} -E[r_{pfo}]\right\vert_-$ where $ \left\vert a\right\vert_- = \max[0,-a]$

+ Qunatile-based risk measures 
    + Value-at-risk(Var)
    + Conditional value-at-risk(CVar)