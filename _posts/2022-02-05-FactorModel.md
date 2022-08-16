---

category: [FinancialEngineering]
title : "Factor Model"
excerpt : 
date: 2022-02-05
use_math : true
---

# __Input of M-V Model__

+ Suppose there are n stocks, then m-v model requires "n mean values, n variance, $\frac {n(n-1)}2$ covariances"
Total of $ 2n + \frac  {n(n-1)}2 $ parameters are required

+ When n is large, this is a very large set of required values

# __Factor Models__

+ Factor : randomness of n stocks can be traced back to a smaller number of basic sources of randomness that influence the individuaol returns

+ Factor model leads to simplified structure for the covariance among assets

+ For commom stocks, the factors might be the size of a company, GDP, unemployment rate, etc.

## __Single Factor Model__
$ 
y_i = a_i + b_if + e_i
$

+ $e_i$ : Error with $ E(e_i) = 0 $

+ Why Factor Model is better?

    + we know the value of $a_i$, and $b_i$, the model can be used to estimate $y_i$
    + Without a factor model, the hours studied will be have to be estimated every time.

+ The $b_i$'s are termed factor loadings because they measure the sensitivity of the return to the factor

+ Single-factor model can be viewed grapgically as defining a linear fit

+ If we know the model, we can predict that the data points will fall near the line

$
r_i = a_i + b_if + e_i
\\
E(stock\;  i)  = a_i + b_i\overline f
\\
var(stock\; i) = b_i^2\sigma_f^2 + \sigma_{e_i}^2
\\
cov(stock\; i, stock\; j ) = b_ib_j\sigma_f^2

$
+ By using a single factor model, we just need a total 3n+2 parameters  

# __Single-Factor Model : Portfolios__

$
r = \sum_{i=1}^nw_ia_i + \sum_{i=1}^nw_ib_if + \sum_{i=1}^nw_ie_i
$

## Diversification
+ let's assume that $var(e_i) = s^2 \; for \; all \; i$. Then for an equally-weighteed portfolio, $w_i=\frac1n$.

$
\sigma_e^2 = \frac1ns^2
$
Hence, $\lim_{n\rightarrow \infty}\sigma_e^2 = 0 $

+ So in a well diversified portfolio, the error term in the factor equation is small.

# __CAPM as a Factor Model__

__Single factormodel__

$r_i = a_i + b_if + e_i$

+ Assume that the single factor is the market rate of return $r_M$
+ We can also express the returns as excess returns 

$r_i -r_f= a_i + b_i(r_M - r_f) + e_i$

+ Single Factor Model is similar to _CAPM_, except _CAPM_ predicst that $\alpha_i = 0 $
+ CAPM is not derived from a factor model
+ Single Factor model is more general than CAPM, because it allows $\alpha$ to be nonzero




=======
---

category: [FinancialEngineering]
title : "Factor Model"
excerpt : 
date: 2022-02-04
use_math : true
---

# __Input of M-V Model__

+ Suppose there are n stocks, then m-v model requires "n mean values, n variance, $\frac {n(n-1)}2$ covariances"
Total of $ 2n + \frac  {n(n-1)}2 $ parameters are required

+ When n is large, this is a very large set of required values

# __Factor Models__

+ Factor : randomness of n stocks can be traced back to a smaller number of basic sources of randomness that influence the individuaol returns

+ Factor model leads to simplified structure for the covariance among assets

+ For commom stocks, the factors might be the size of a company, GDP, unemployment rate, etc.

## __Single Factor Model__
$ 
y_i = a_i + b_if + e_i
$

+ $e_i$ : Error with $ E(e_i) = 0 $

+ Why Factor Model is better?

    + we know the value of $a_i$, and $b_i$, the model can be used to estimate $y_i$
    + Without a factor model, the hours studied will be have to be estimated every time.

+ The $b_i$'s are termed factor loadings because they measure the sensitivity of the return to the factor

+ Single-factor model can be viewed grapgically as defining a linear fit

+ If we know the model, we can predict that the data points will fall near the line

$
r_i = a_i + b_if + e_i
\\
E(stock\;  i)  = a_i + b_i\overline f
\\
var(stock\; i) = b_i^2\sigma_f^2 + \sigma_{e_i}^2
\\
cov(stock\; i, stock\; j ) = b_ib_j\sigma_f^2

$
+ By using a single factor model, we just need a total 3n+2 parameters  

# __Single-Factor Model : Portfolios__

$
r = \sum_{i=1}^nw_ia_i + \sum_{i=1}^nw_ib_if + \sum_{i=1}^nw_ie_i
$

## Diversification
+ let's assume that $var(e_i) = s^2 \; for \; all \; i$. Then for an equally-weighteed portfolio, $w_i=\frac1n$.

$
\sigma_e^2 = \frac1ns^2
$
Hence, $\lim_{n\rightarrow \infty}\sigma_e^2 = 0 $

+ So in a well diversified portfolio, the error term in the factor equation is small.

# __CAPM as a Factor Model__

__Single factormodel__

$r_i = a_i + b_if + e_i$

+ Assume that the single factor is the market rate of return $r_M$
+ We can also express the returns as excess returns 

$r_i -r_f= a_i + b_i(r_M - r_f) + e_i$

+ Single Factor Model is similar to _CAPM_, except _CAPM_ predicst that $\alpha_i = 0 $
+ CAPM is not derived from a factor model
+ Single Factor model is more general than CAPM, because it allows $\alpha$ to be nonzero




