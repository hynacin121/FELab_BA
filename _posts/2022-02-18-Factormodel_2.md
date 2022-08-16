---
category: [FinancialDataAnalysis]
title : "Factor Models PCA"
excerpt : ""

date: 2022-02-18
use_math : true
---

# __Principal Components Analysis(PCA)__

+ PCA is a statical approach for finding a factor model
+ PCA is commonly used to compute the large-dimensional covariance matrices and to reduce dimensionality
+ Advantage : PCA finds orthogonal factors
+ Disadvantage : Interpretation of factors is not easy
+ Based on eigenvalue-eigenvector decomposition.
    + PCA can be applied to the correlation matrix 
+ X : set of n returns with time series data
    + each column of X is a time-series data of an asset
    + PCA describes variation relative to the mean
+ V : sample covariance matrix of X
+ T is the number of data points used in the analysis and n is the number of variables

## __Sample Covariance__

$ 
V = \frac1{T-1} X^TX
$

## __Sample and Asset sizes__

+ Sample size T is collected to be larger than the number of assets n.
+ If T < n, there will be one or more zero eigen values

## __Principal Components__

+ Each PC is defined to be a linear combination of columns of X
    + PCs are uncorrelated with each other
    + first PC explains the most variation
+ There are several ways to find the PC
    + Maximize variance
    + Minimize projection residuals
    + Eigen decomposition
    + Singular value decomposition

+ Denote by $\Lambda$ the diagonal matrix of eigenvalues of V and by W the orthogonal matrix of eigenvectors of V
+ In $\Lambda$ we order the eigenvalues from largest to smallest
    
    $ \lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_n $
+ The matrix of principal components of V is a T x n matrix P defined as a product of matrix of input data X with eigenvector matrix of V : P = XW

+ The $m^{th}$ principal component of V is defined as the $m^{th}$ column of P.
$
p_m = \omega_{1m}x_1 + \omega_{2m}x_2 + \dots \omega_{km}x_k 
$
where $ w_m = (\omega_{1m}, \omega_{2m}, \dots, \omega_{km})'$ is the eigen vector corresponding to $\lambda_m$, the  $m^{th}$ largest eigenvalue of V. That is $w_m$ is the  $m^{th}$ column of W. Thus  $p_1$ belongs to the first and largest eigenvalue  $\lambda_1$

+ The covariance matrix $T^{-1}P'P$ of the principal components is $\Lambda$.
$
T^{-1}P'P = T^{-1}W'X'XW = W'VW = W'W\Lambda = W^{-1}W\Lambda = \Lambda \\ \frac1TW'X'XW = W'(\frac1TX'X)W, \quad where\; \frac1TX'X \; is\; covariance\; matrix  
$
+ This shows that
    + the covarinace matrix is diagonal, PCs are uncorrelated
    + $m^{th}$ PC is $\lambda_m$, the $m^{th}$ largest eigenvalue of V 

+ The proportion of this total variaction that is explained by  $m^{th}$ principal component is
$
\frac{\lambda_m}{\lambda_1 + \lambda_2 + \dots + \lambda_n}
$

## __PC Representation__

+ Since $P = XW, \; W' = W^{-1}$ we have $X = PW'$
+ In other words, we can write each of the original returns that are input to the PCA as linear combination of the principal components as 
$
x_i = \omega_{i1}p_1 + \omega_{i2}p_2 +\dots + \omega_{ik}p_k
$
A representation using the first three principal components is 
$
x_i  \approx \omega_{i1}p_1 + \omega_{i2}p_2 + \omega_{i3}p_3
$
+ In matrix notation.
$
X \approx P^*{W^*}^T
$
+ PCA is useful for reducing dimensions in highly correlated systems where n is very large and $ k \ll n $


# __PCA in Python__

```python
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
```

+ Using Scikit


```python
X = np.array([[-1, -1], [-2, -2], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components = 2)
pca.fit(X)
print(pca.explained_variance_ratio_)

```

    [0.99157153 0.00842847]
    

+ Using Covariance 


```python
df = pd.DataFrame(X)

covariance = df.cov()
eigenvalue = np.linalg.eigvals(covariance)

pc1 = np.linalg.eigvals(df.cov())[0]/sum(np.linalg.eigvals(df.cov()))
pc2 = np.linalg.eigvals(df.cov())[1]/sum(np.linalg.eigvals(df.cov()))

print(pc1, pc2)
```

    0.9915715292774702 0.008428470722529814
    


