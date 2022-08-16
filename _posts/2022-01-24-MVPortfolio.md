---

category: [FinancialEngineering]
title : "Mean Variance Portfolio Theory"
excerpt : "Mean Variance Portfolio and Efficient Frontier"

date: 2022-01-24
use_math : true
---

# __Portfolio Risk__

+ Portfolio with only a few assets = high degree of risk
+ Risk of the return of portfolio can be reduced by   diversification
+ Effect of Diversification can be explaned by the mean-variance framework

$
Var = \sum _{i=1} ^n \sum _{j=1} ^n (w_iw_j\sigma_{ij})
$

Suppose each asset has same rate of return with mean _m_ and variance σ.\
Each return pair has a same covariance.\
Then, the variance becomes
$
var(r) = \frac {0.7\sigma^2}n + 0.3\sigma^2
$

+ As n increases, variance decreases.




# __Diagram of a Portfolio__
<p align = "center">
<img src = "/assets/img/DiagramOfPortfolio.jpg" , height="300x", width="300px" >
</p>

+ Combination of assets 1 and 2 trace out a curve that includes the two assets

+ Solid portion corresponds to positive combinations of the two assets

# __Mean-Variance Set__
## __Feasible Set__

<p align = "center">
<img src = "/assets/img/FeasibleSet.jpg" , height="300x", width="800px" >
</p>

+ Convex to left
+ Left boundary of a feasible set is called __minimum-varinace set__
+ There is Special point on this set having minimum variance -> global minimum-variance portfolio
+ Upper portion of the minimum-variance set is termed the __efficient frontier__ 


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import optimize
from datetime import datetime
%matplotlib inline

try:
    import pandas_datareader.data as web
except:
    %pip install pip pandas-datareader
    import pandas_datareader.data as web
  
```


```python
def Asset(ticker , start, end):
    return(web.get_data_yahoo(ticker, start, end)['Adj Close'])
```


```python
data = Asset(['AAPL','GOOGL', 'MSFT', 'AMZN'], '2011-01-01', '2021-12-31')
ret = data.pct_change().dropna()
data = data/data.iloc[0]*100

data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>AAPL</th>
      <th>GOOGL</th>
      <th>MSFT</th>
      <th>AMZN</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-12-31</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>2011-01-03</th>
      <td>102.173256</td>
      <td>101.747561</td>
      <td>100.250818</td>
      <td>102.344445</td>
    </tr>
    <tr>
      <th>2011-01-04</th>
      <td>102.706471</td>
      <td>101.372120</td>
      <td>100.644935</td>
      <td>102.783330</td>
    </tr>
    <tr>
      <th>2011-01-05</th>
      <td>103.546638</td>
      <td>102.542215</td>
      <td>100.322472</td>
      <td>104.122221</td>
    </tr>
    <tr>
      <th>2011-01-06</th>
      <td>103.462933</td>
      <td>103.288047</td>
      <td>103.260478</td>
      <td>103.255556</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-12-27</th>
      <td>1825.785237</td>
      <td>995.057562</td>
      <td>1556.170970</td>
      <td>1885.216607</td>
    </tr>
    <tr>
      <th>2021-12-28</th>
      <td>1815.255473</td>
      <td>986.853275</td>
      <td>1550.717841</td>
      <td>1896.233317</td>
    </tr>
    <tr>
      <th>2021-12-29</th>
      <td>1816.166813</td>
      <td>986.638027</td>
      <td>1553.898856</td>
      <td>1880.011122</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>1804.219599</td>
      <td>983.580298</td>
      <td>1541.947514</td>
      <td>1873.827718</td>
    </tr>
    <tr>
      <th>2021-12-31</th>
      <td>1797.841150</td>
      <td>974.508122</td>
      <td>1528.314829</td>
      <td>1852.411160</td>
    </tr>
  </tbody>
</table>
<p>2770 rows × 4 columns</p>
</div>




```python
n = len(data.columns)
returns = []
stds = []
wgt = []
count = 5000
```


```python
for i in range(count):
    weights = np.random.random(n)
    weights /= sum(weights)
    wgt.append(weights)
    mean = np.sum(weights * ret.mean()*252)
    var = np.dot(weights.T, np.dot(ret.cov()*252, weights))
    cov = np.sqrt(var)    
    returns.append(mean)
    stds.append(cov)
    
```


```python
wgt2 = pd.DataFrame(wgt, columns = data.columns)
dt = {'Returns':returns, 'Stds': stds}
dt = pd.DataFrame(dt)

eff = pd.concat([dt, wgt2], axis = 1)

eff
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Returns</th>
      <th>Stds</th>
      <th>AAPL</th>
      <th>GOOGL</th>
      <th>MSFT</th>
      <th>AMZN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.275116</td>
      <td>0.226846</td>
      <td>0.382664</td>
      <td>0.440495</td>
      <td>0.057298</td>
      <td>0.119544</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.272488</td>
      <td>0.225754</td>
      <td>0.023177</td>
      <td>0.357022</td>
      <td>0.437999</td>
      <td>0.181801</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.272332</td>
      <td>0.230374</td>
      <td>0.034066</td>
      <td>0.457029</td>
      <td>0.218236</td>
      <td>0.290669</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.305676</td>
      <td>0.246441</td>
      <td>0.407604</td>
      <td>0.012275</td>
      <td>0.080313</td>
      <td>0.499807</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.279730</td>
      <td>0.233068</td>
      <td>0.543789</td>
      <td>0.351188</td>
      <td>0.063686</td>
      <td>0.041337</td>
    </tr>


  </tbody>
</table>
<p>5000 rows × 6 columns</p>
</div>




```python
min_vol_port = eff.iloc[eff['Stds'].idxmin()]

plt.subplots(figsize=[6,6])
plt.scatter(eff['Stds'], eff['Returns'], marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=150)


#https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/
```
<p>
<img src = "/assets/img/EfficientFrontier.png" , height="357x", width="388px" >
</p>
    


# __MarKowitz Model__

## Assumptions
+ All investor only care about mean and variacnce
+ All investors are rational
+ To find a minimum-variance portfolio, we fix the mean value at some arbitrary value r
+ Then we find the feasible portfolio of minimum variance than has this mean

$$
mimimize_w \frac12 \sum_{i=1}^n \sum_{j=1}^nw_iw_j\sigma_{ij} \\
subject\; to \sum_{i=1}^nw_ir_i = r \\
\sum_{i=1}^nw_i = 1
$$

+ It turns out that solving the original problem is the same as solving the set of equations on the below

$$
\sum_{i=1}^n\sigma_{ij} - \lambda r_i - \mu = 0  \quad
for \; i = 1, 2, ..., n\\ 
\sum_{i=1}^nw_ir_i = r \\
\sum_{i=1}^nw_i = 1

$$

# __One Fund Theorem__

+ There is a single portfolio F of risk assets such that any efficient portfolio can be constructeed as a combination of the fund F and the risk-free asset

<p>
<img src = "/assets/img/OneFundTheorem.jpg" , height="357x", width="388px" >
</p>
=======
---

category: [FinancialEngineering]
title : "Mean Variance Portfolio Theory"
excerpt : 

date: 2022-01-18
use_math : true
---

# __Portfolio Risk__

+ Portfolio with only a few assets = high degree of risk
+ Risk of the return of portfolio can be reduced by   diversification
+ Effect of Diversification can be explaned by the mean-variance framework

$
Var = \sum _{i=1} ^n \sum _{j=1} ^n (w_iw_j\sigma_{ij})
$

Suppose each asset has same rate of return with mean _m_ and variance σ.\
Each return pair has a same covariance.\
Then, the variance becomes
$
var(r) = \frac {0.7\sigma^2}n + 0.3\sigma^2
$

+ As n increases, variance decreases.




# __Diagram of a Portfolio__
<p align = "center">
<img src = "/assets/img/DiagramOfPortfolio.jpg" , height="300x", width="300px" >
</p>

+ Combination of assets 1 and 2 trace out a curve that includes the two assets

+ Solid portion corresponds to positive combinations of the two assets

# __Mean-Variance Set__
## __Feasible Set__

<p align = "center">
<img src = "/assets/img/FeasibleSet.jpg" , height="300x", width="800px" >
</p>

+ Convex to left
+ Left boundary of a feasible set is called __minimum-varinace set__
+ There is Special point on this set having minimum variance -> global minimum-variance portfolio
+ Upper portion of the minimum-variance set is termed the __efficient frontier__ 


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import optimize
from datetime import datetime
%matplotlib inline

try:
    import pandas_datareader.data as web
except:
    %pip install pip pandas-datareader
    import pandas_datareader.data as web
  
```


```python
def Asset(ticker , start, end):
    return(web.get_data_yahoo(ticker, start, end)['Adj Close'])
```


```python
data = Asset(['AAPL','GOOGL', 'MSFT', 'AMZN'], '2011-01-01', '2021-12-31')
ret = data.pct_change().dropna()
data = data/data.iloc[0]*100

data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>AAPL</th>
      <th>GOOGL</th>
      <th>MSFT</th>
      <th>AMZN</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-12-31</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>2011-01-03</th>
      <td>102.173256</td>
      <td>101.747561</td>
      <td>100.250818</td>
      <td>102.344445</td>
    </tr>
    <tr>
      <th>2011-01-04</th>
      <td>102.706471</td>
      <td>101.372120</td>
      <td>100.644935</td>
      <td>102.783330</td>
    </tr>
    <tr>
      <th>2011-01-05</th>
      <td>103.546638</td>
      <td>102.542215</td>
      <td>100.322472</td>
      <td>104.122221</td>
    </tr>
    <tr>
      <th>2011-01-06</th>
      <td>103.462933</td>
      <td>103.288047</td>
      <td>103.260478</td>
      <td>103.255556</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-12-27</th>
      <td>1825.785237</td>
      <td>995.057562</td>
      <td>1556.170970</td>
      <td>1885.216607</td>
    </tr>
    <tr>
      <th>2021-12-28</th>
      <td>1815.255473</td>
      <td>986.853275</td>
      <td>1550.717841</td>
      <td>1896.233317</td>
    </tr>
    <tr>
      <th>2021-12-29</th>
      <td>1816.166813</td>
      <td>986.638027</td>
      <td>1553.898856</td>
      <td>1880.011122</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>1804.219599</td>
      <td>983.580298</td>
      <td>1541.947514</td>
      <td>1873.827718</td>
    </tr>
    <tr>
      <th>2021-12-31</th>
      <td>1797.841150</td>
      <td>974.508122</td>
      <td>1528.314829</td>
      <td>1852.411160</td>
    </tr>
  </tbody>
</table>
<p>2770 rows × 4 columns</p>
</div>




```python
n = len(data.columns)
returns = []
stds = []
wgt = []
count = 5000
```


```python
for i in range(count):
    weights = np.random.random(n)
    weights /= sum(weights)
    wgt.append(weights)
    mean = np.sum(weights * ret.mean()*252)
    var = np.dot(weights.T, np.dot(ret.cov()*252, weights))
    cov = np.sqrt(var)    
    returns.append(mean)
    stds.append(cov)
    
```


```python
wgt2 = pd.DataFrame(wgt, columns = data.columns)
dt = {'Returns':returns, 'Stds': stds}
dt = pd.DataFrame(dt)

eff = pd.concat([dt, wgt2], axis = 1)

eff
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Returns</th>
      <th>Stds</th>
      <th>AAPL</th>
      <th>GOOGL</th>
      <th>MSFT</th>
      <th>AMZN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.275116</td>
      <td>0.226846</td>
      <td>0.382664</td>
      <td>0.440495</td>
      <td>0.057298</td>
      <td>0.119544</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.272488</td>
      <td>0.225754</td>
      <td>0.023177</td>
      <td>0.357022</td>
      <td>0.437999</td>
      <td>0.181801</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.272332</td>
      <td>0.230374</td>
      <td>0.034066</td>
      <td>0.457029</td>
      <td>0.218236</td>
      <td>0.290669</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.305676</td>
      <td>0.246441</td>
      <td>0.407604</td>
      <td>0.012275</td>
      <td>0.080313</td>
      <td>0.499807</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.279730</td>
      <td>0.233068</td>
      <td>0.543789</td>
      <td>0.351188</td>
      <td>0.063686</td>
      <td>0.041337</td>
    </tr>


  </tbody>
</table>
<p>5000 rows × 6 columns</p>
</div>




```python
min_vol_port = eff.iloc[eff['Stds'].idxmin()]

plt.subplots(figsize=[6,6])
plt.scatter(eff['Stds'], eff['Returns'], marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=150)


#https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/
```
<p>
<img src = "/assets/img/EfficientFrontier.png" , height="357x", width="388px" >
</p>
    


# __MarKowitz Model__

## Assumptions
+ All investor only care about mean and variacnce
+ All investors are rational
+ To find a minimum-variance portfolio, we fix the mean value at some arbitrary value r
+ Then we find the feasible portfolio of minimum variance than has this mean

$$
mimimize_w \frac12 \sum_{i=1}^n \sum_{j=1}^nw_iw_j\sigma_{ij} \\
subject\; to \sum_{i=1}^nw_ir_i = r \\
\sum_{i=1}^nw_i = 1
$$

+ It turns out that solving the original problem is the same as solving the set of equations on the below

$$
\sum_{i=1}^n\sigma_{ij} - \lambda r_i - \mu = 0  \quad
for \; i = 1, 2, ..., n\\ 
\sum_{i=1}^nw_ir_i = r \\
\sum_{i=1}^nw_i = 1

$$

# __One Fund Theorem__

+ There is a single portfolio F of risk assets such that any efficient portfolio can be constructeed as a combination of the fund F and the risk-free asset

<p>
<img src = "/assets/img/OneFundTheorem.jpg" , height="357x", width="388px" >
</p>
>>>>>>> 3ede3443c43a311f1c71a79516f8e72afd402a54
    