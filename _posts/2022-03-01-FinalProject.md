---
category: [FinancialDataAnalysis]
title : "Portfolio for Great Lockdown"
excerpt : ""

date: 2022-03-01
use_math : true
mathjax : true
---
__SWCON424-01 금융데이터 분석 기말프로젝트__

2015103604 한지운 

2014102714 엄세웅

2017103747 정희재


# __주제__

+ IMF는 2020년을 Great Lockdown이라고 말하면서 Great Depression 이후 최악의 경기침체라고 말하였다. 2020년 3월, 코스피, S&P 500, 닛케이 지수가 20% 이하로 떨어지는 사태가 발생하였다. 이러한 경제 침체를 겪으며 세계적인 공황을 방어할 수 있는 포트폴리오를 구성해보고자 주제를 선정하였다.

# __All Weather Portfolio__

## __All Weather Portfolio__

+ All Weather Portfolio는 각 자산군의 상관관계가 경제 환경에 따라 달라지지만, 각각의 상황에서는 일관성을 갖는다는 사실을 기반하여 투자를 한다. 인플레이션에 대한 기대감이 높을 때는 주식과 채권의 상관관계가 높아지지만 경제성장에 기대감이 높을 때는 반대로 낮아지게 된다. 

    경제성장률과 인플레이션이 기대보다 높고 낮음에 따라 4개 상황에 동일한 리스크로 투자할 수 있게 포트폴리오를 구성한다. 4개의 상황에 유리한 자산은 아래 그림과 같다.
<br>
<br/>

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>AAPL</th>
      <th>GOOGL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rising</th>
      <td>주식, 회사채, 원자재, 이머징국가 채권</td>
      <td>물가연동채권, 원자재, 이머징국가 채권</td>
    </tr>  
    <tr>
      <th>Falling</th>
      <td>국채, 물가연동채권</td>
      <td>주식, 국채</td>
  </tbody>
</table>
</div>
<br>
<br/>
     경제를 예측하지 않고 각각의 상황에 들어갈 확률을 25%로 가정하여 자산을 배분한다. 자산군 내에서도 리스크를 헷지하기 위하여 다양한 자산에 투자할 수 있도록 구성하였다.
<br>
<br/>


## __Risk Parity Model__

+ 기존 Mean-Variance Model은 공분산, 기대 수익률, 위험회피계수를 활용하여 모델을 구성한다. 이 때 기대수익률은 추정 에러가 높고 신뢰도가 낮다. M-V모델은 이러한 기대수익률에 기반한 모델이기 때문에 불안정하다. 반면 Risk Parity  Model은 위험기여도로 동일하도록 구성한다. 이때 공분산 행렬만 사용하기 때문에 M-V모델보다 안정성을 갖는다.

  Risk Parity Model은 Marginal Risk Contribution과 Risk Contribution을 활용하여 구성한다. Marginal Contribution은 특정 종목의 비중을 한 단위 늘렸을 때 증가하는 포트폴리오의 변동성(1)이다. Risk Contribution은 개별종목이 전체 포트폴리오의 위험에 기여하는 정도로 MRC에 가중치를 곱하는 값(2)로 나타낸다. 

$
MRC = \frac{\partial\sigma_p}{\partial w_i}\\
\\
RC = w_i \times \frac{\partial\sigma_p}{\partial w_i}
$
+ Python으로 Risk Contribution을 단순하게 계산하기 위해 한가지 계산 과정이 추가로 필요하다.

$
RC = w_i \times \frac{\partial\sigma_p}{\partial w_i} \\
$
$
pf)\quad  (\sqrt{f(x)})' = \frac{f'(x)}{2\sqrt{f(x)}} \\
MRC :\quad \frac{\partial\sigma_p}{\partial w_i} = \frac{2\Sigma w}{2\sqrt{w'\Sigma w}} \\
RC : \quad w_i \times \frac{\partial\sigma_p}{\partial w_i} = \frac{2w' \Sigma w}{2\sqrt{w'\Sigma w}} = \sqrt{w'\Sigma w} = \sigma_p
$

$
Equal \; Risk \; Contribution =  \frac{\sigma_p}n
$

+ 모델에서 동일한 Risk Contribution을 주기 위해서 표준편차를 자산의 개수로 나눈 값을 활용한다. Risk Parity Model는 최소화 모델로 구축할 수 있다.
<br>
<br/>
$
\min \sum \sum(w_iMRC_i - w_jMRC_j)^2 \\
s.t \sum w_i = 1\\
w_i \; > \; 0 \; for \; \forall i 
$
$
\min \sum[w_i - \frac{\sigma(w)^2}{(\Sigma w)_iN }]^2 \\
s.t \sum w_i = 1\\
w_i \; > \; 0 \; for \; \forall i 
$
<br>
<br/>
+ Risk Parity Model은 여러가지 최소화 모델로 구축할 수 있다. 좌측의 모델의 경우 i번째 자산의 RC가 j번째 자산의 RC와 같을 때 minimize가 되는 형식으로 모든 자산의 RC가 동일 해진다. 우측의 모델의 경우 $w_i-\frac{σ(w)^2}{((Σw)_i N)}= 0$일 때 최소화된다. 이때 i번째 자산의 분산은 $w_i (Σw)_i$ 로 $w_i (Σw)_i=σ(w)^2/N $일 때, 즉 각 자산의 분산이 전체 분산을 자산으로 나눈 값으로 동일 할 때 최소화된다.

  이 모델을 활용함으로써 특정 경우에 따라는 자산의 RC가 동일하게 나오지 않을 상황이 생길 수 있다. 이 때 좌측모델과의 결과값의 차이가 발생할 수 있다. 그럼에도 이 프로젝트에서는 코딩에 용이하여 따라 좌측 모델을 활용할 것이다.
<br>
<br/>
## __All Weather by Risk Parity__

+ Risk Parity Model을 All Weather Portfolio에 적용하기 위해서 포트폴리오를 2번 구축해야한다. 첫째로 그림1에서 나뉜 대로 총 4개의 포트폴리오를 구성한다. 이 때 포트폴리오는 동일가중치를 활용한다. 이 후 Risk Parity를 활용하기 때문에 4개의 포트폴리오는 동일가중치를 활용해도 괜찮다고 가정한다. 4개의 포트폴리오를 하나의 자산으로 보고 Risk Parity 모델을 활용하여 가중치를 구한다.  

# __비교군__ 

+ All Weather Portfolio와의 기대 수익률, Drawdown을 비교하기 위한 벤치마크로 두 가지 포트폴리오를 사용하였다. GMV Portfolio는 수익률을 일부 포기하는 대신 압도적인 안정성을 보장하는 모델이며, 60/40 Portfolio는 안정성을 기반으로 꾸준히 높은 수익률을 보여주는 모델이다. 두 모델을 벤치마킹하여 All Weather Portfolio의 상대적인 수익률과 안정성을 비교하고 포트폴리오의 성능을 평가하고자 하였다.

<br>
</br>

## __60/40__

+ 먼저 첫 번째 벤치마크로 GMV Portfolio(최소 분산 포트폴리오)를 사용하였다. GMV Portfolio는 포트폴리오의 분산을 최소화함으로써 분산을 통해서 상쇄되지 않는 리스크를 최대한 낮춰 포트폴리오의 안정성을 최대화한다는 장점을 가지고 있다. 수익률은 낮다는 단점을 가지고 있지만 포트폴리오의 리스크가 가장 낮은 모델이기 때문에 코로나 대유행으로 변동성이 높은 주식 시장에서도 안정적인 모습을 보여줄 것이라고 생각했다. 따라서 All Weather Portfolio와의 drawdown 차이 대비 수익률을 비교하고자 벤치마크 모델로 선정하였다.

## __GMV__
+ 두 번째 벤치마크로는 60/40 Portfolio를 사용하였다. 60/40 Portfolio는 주식의 비중을 60%, 채권의 비중을 40%를 두고 투자하는 conventional portfolio이다. 해당 포트폴리오는 주식 시장의 변동성을 채권을 통해 리스크를 완충 및 분산시키면서도 높은 수익률을 낸다는 장점을 가지고 있다. 특히, 1983년부터 2010년까지 S&P 500지수는 연간 11.2%, 채권 인덱스 지수는 연간 7% 상승하여 평균 연간 9.9%의 높은 수익률을 보여주었다. 주식에 60%의 높은 비중을 두고 있음에도 불구하고 물가연동체 TIPS와 금을 통해 포트폴리오에 안정성을 추가하여 과거 2007년의 주식 폭락에도 포트폴리오 수익률을 방어할 수 있었다. 이렇게 높은 수익률과 안정성을 가지고 있는 60/40 Portfolio를 벤치마킹하여 2020년 코로나 대유행 기간 동안의 All Weather Portfolio와의 연간 수익률과 drawdown을 비교하고자 하였다.

# __코드 설명__
 해당 코드를 편리하게 실행하기 위해서는 Docker가 필요하며, 웹사이트는 Chrome에 최적화 되어있으니 Chrome에서 전체화면이 가장 용이하다.

링크 : https://github.com/lydiahjchung/2020_FDA 


## __Backtest 구현__
+ 크게 두 개의 클래스로 구현이 되어 있고, 세부적으로 나누어진 Portfolio의 개별 클래스들은 이 두 클래스를 바탕으로 생성했다. 데이터 시각화는 전부 flask의 app.py에 구현되어 있다. Plotly를 사용하여 구현하였고 backtest의 데이터를 가지고 단순히 시각화해주기 때문에 설명은 제외한다.

  +	 __Asset__

      ticker, start date, end date를 입력 받아 데이터를 해당 ticker의 해당 기간 동안의 데이터를 저장해두는 객체이다. Adjusted Close 데이터를 가져와 percentage change를 진행해 준 데이터를 저장해둔다. 이후 Portfolio에서 이 asset 클래스를 활용하여 포트폴리오 구성 시 데이터 활용이 가능하며 모든 데이터는 Yahoo Finance 데이터를 활용했다.

   + __Portfolio__

      Portfolio 구성으로 backtest 및 시각화 구현을 위해 데이터 처리를 해주는 클래스이다. 아래의 4가지 함수들이 주요한 계산을 구현하고, 나머지는 모두 getter 함수이다.

  +	__backtest()__

      사용자에게 받은 initial balance, rebalancing 주기, look back 주기를 바탕으로 실행된다. 처음 backtest 를 시작하면 사용자에게 받은 initial balance를 초기 weight와 asset으로 맞추어 나누어줘야 한다. 60/40의 경우는 고정 weight이지만, GMV와 All Weather의 경우는 아니기에 backtest 시작과 동시에 start date를 기준으로 사용자의 look back기간만큼의 데이터를 통하여 초기 weight를 계산해준다. 이 weight들과 60/40의 고정 웨이트를 통해 start date의 balance를 asset 별로 나누어준다.

      이후 rebalancing 주기마다 look back 기간만큼의 데이터를 가지고 weight를 재설정해준 값으로 balance를 다시 나누어준다. rebalancing이 아닌 기간 동안은 설정된 weight와 asset별 데이터를 가지고 balance에 대한 데이터를 계속 쌓아준다. 

      end date까지 동일한 과정이 이어지고, 이를 하나의 data frame로 저장한다. Portfolio에 해당하는 Total balance값과 각 asset별 balance가 저장되므로 column으로 Total 과 각 asset 들이 있다.

  +	__backtest_result() & balance_result()__

    backtest_result 함수는 backtest 함수 구현을 통해 저장한 backtest 결과의 데이터를 가지고 rate of return, cumulative return, CAGR, drawdown, MDD 의 값을 구하기 위하여 사용된다. balance_result는 이 backtest_result 함수 내에서 호출된다. Backtest의 결과로 각 column 별로 balance가 날짜에 따라 어떻게 변하는 지가 들어있고, 현재 backtest의 결과에 대한 지표 값들을 계산하고 싶은 것이기에 Total balance 열에 대한 데이터만 balance_result 함수를 통하여 각 지표들의 값을 구해준다.

    balance_result 함수에서는 날짜별로 저장된 변화되는 balance 값을 통하여 rate of return, cumulative return, CAGR, drawdown, MDD, standard deviation, sharpe ratio 값을 구해준다. backtest_result에서는 이 값 중 4 가지만 저장하지만, 향후 다른 값들은 summary를 통해 사용되도록 구현했다.

  +	__summarize()__

    Portfolio의 전체 summary를 출력해준다. 이를 통해 initial balance, portfolio의 final balance, CAGR, MDD, sharpe ratio의 값을 출력해준다.
  + __periodic_result(mode)__

    Annual return, monthly return의 값을 향후 시각화해주기 위하여 구현한 함수이다. Mode를 통하여 annual, monthly return 중 한 가지를 골라 데이터를 출력할 수 있다. backtest를 통해 구한 total balance의 데이터를 가지고 각 year / month 별로 의 balance 데이터를 통해 하나의 data frame에 저장해준다.


## __Portfolio 구현__

+ 모든 포트폴리오는 3번에서의 두 클래스를 바탕으로 구현이 되기에, 다른 점은 rebalancing 주기에 맞춰 weight를 다시 구하는 과정에만 차이가 존재한다.

  + 60/40

    60/40 포트폴리오의 경우 주식 60%, 채권 40% 의 고정 비율로 backtest가 진행된다. 따라서, rebalancing 또한 고정된 weight로 진행되어 복잡한 weight 계산 과정이 따로 없다. 하지만 input을 받는 check box들에서 볼 수 있듯이, 총 6가지의 자산군으로 나누어 입력을 받기에, 60/40의 경우에는 선택된 input 중 Nominal Bond 와 Global Equities에서 선택된 자산들만을 고려한다. 한 자산군당 여러 input을 받을 수 있기에, 각 bond와 equities를 40%와 60%의 비율로 가져가되, 각 자산군 안에서의 개수에 맞게 그 비율을 조정해준다.

    예를 들어, Nominal Bond에서 총 2개를 선택하고, Global Equities에서 총 3개를 선택한 경우, bond는 40%, equities는 60%의 비율을 유지해야하기에 Nominal Bond 내의 각 자산은 20%씩의 비율을 가지게 되고, Global Equities 내의 자산 또한 각 20%씩의 비율을 가지고 portfolio의 계산이 진행된다. 

  +	GMV

    먼저 Asset 클래스를 상속받는 GMV_Asset 클래스에 주식 리스트와 시작 일, 마지막 일을 input으로 사용하여 주어진 포트폴리오의 데이터 프레임을 얻는다. 이후 각 주식의 adjusted close price에 percentage change를 하고 numpy의 mean()과 cov() 모듈을 사용하여 포트폴리오의 평균 수익률과 공분산 데이터 프레임을 만든다. 이렇게 얻은 평균 수익률과 공분산 데이터프레임을 GMVPortfolio 클래스에 넣고 각각을 연 단위 수치로 변환해 준 다음, scipy의 optimize.minimize 모듈을 통해 포트폴리오의 최소 분산을 만족하는 weight 값을 얻는다. 이 과정을 rebalancing 기간마다 실행하여 dynamic weight allocation을 구현했다.

  +	All Weather

    GMV의 경우 주어진 포트폴리오를 주식, 채권, 회사채, 원자재 등으로 구분하지 않고 하나로 묶어서 관리하지만 All Weather Portfolio의 경우 각각을 구분해서 계산하기 때문에 각 카테고리를 dictionary 형태로 구분했다. Asset 클래스에서 상속받는 AWF_Asset 클래스에 주식 리스트와 시작 일, 마지막 일을 input으로 사용하여 각 자산의 수익률을 나타내는 dataframe을 dictionary 형태로 저장하여 반환한다. 이후 AllWeatherPortfolio 클래스를 통해 각 카테고리를 growth, rising, emerging, falling에 따라 4개로 구분한다. 이렇게 구분한 데이터를 RiskPartiy 함수에 넣고 scipy의 optimize.minimize를 통해 카테고리 4개에 대한 weight 값을 얻는다. 마지막으로 4개의 weight값을 각 카테고리 내의 ticker들에 맞게 한번 더 쪼개어준다. 예를 들어 주식의 개수가 총 4개이고 주식에 대한 weight이 0.2라면 각 주식에 대해 0.05의 weight을 할당한다. 이 과정을 GMV와 마찬가지로 rebalancing 기간마다 실행하여 dynamic weight allocation을 구현했다.

# __Case Study__

+ Case Study를 진행하기 위해 자산을 임의로 지정하였다. 기간은 2008년 2월 1일부터 2020년 10월 30일까지로 설정하였고 Rebalance기간은 6개월 Rebalance 때 lookback 기간은 6개월로 설정하였다.

<br>
<br/>
<div>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Nominal Bond</th>
      <th>iShares 1-3 Year Treasury Bond ETF</th>
    </tr>
    <tr>
      <th>Inflation-linked Bond</th>
      <th>iShares TIPS Bond ETF</th>
    </tr>
    <tr>
      <th>US Corporate Debt Spreads</th>
      <th>iShares iBoxx $ Investment Grade Corporate Bond ETF</th>
    </tr>
    <tr>
      <th>Global Equities</th>
      <th>Vanguard Total Stock Market Index Fund ETF Shares</th>
    </tr>
    <tr>
      <th>EMD Spreads</th>
      <th>iShares MSCI Brazil ETF</th>
    </tr>      
    <tr>
      <th>Commodities</th>
      <th>Invesco DB Commoditiy Index Tracking Fund</th>
    </tr>     
  </thead>
</table>
</div>
<br>
<br/>

<p>
<img src = "/assets/img/FinalProject_Drawdown.png"  >
</p>

+ Drawdown은 위와 같이 나왔다. 예상했던 것처럼 2008~2009년 2020년초에 큰 Drawdown이 있었다. 전반적으로 GMV의 Drawdown이 안정적이었고 60/40의 Drawdown이 가장 불안정 했다.

<p>
<img src = "/assets/img/FinalProject_Drawdown2.png"  >
</p>

+ Recession이었던 2008년을 자세히 살펴보면 All Weather Portfolio가 가장 안정적인 모습을 보였고 GMV와 60/40이 25%가 넘어가는 drawdown을 보였다.

<p>
<img src = "/assets/img/FinalProject_CumulativeReturn.png"  >

<img src = "/assets/img/FinalProject_MonthlyReturn.png"  >
</p>

+ 누적수익률의 경우 60/40, All Weather Portfolio, GMV의 순으로 좋았다. All Weather의 경우 주식의 비중이 20% 아래로 하락하여 60/40에 비해 수익률이 낮음을 알 수 있었다. 

+ Fama-French 5 Factor Model과 회귀분석을 진행한 결과는 다음과 같다. 
<p>
<img src = "/assets/img/FinalProject_FactorModel1.png"  >
<img src = "/assets/img/FinalProject_FactorModel2.png"  >
</p>

+ All Weather Portfolio와 GMV 둘은 alpha 값이 0보다 크고 p값이 유의수준보다 작기 때문에 유의미하다. 이에 따라 두 모델은 Factor Model보다 수익률이 좋다고 할 수 있다. 

세 포트폴리오모델의 정량적 값들을 정리하면 다음과 같다.


<br>
<br/>
<div>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>60/40</th>
      <th>GMV</th>
      <th>AWP</th>
    </tr>
   </thead>
   <tbody> 
    <tr>
      <th>Final Balance</th>
      <td>2473.46</td>
      <td>1579.73</td>
      <td>1837.10</td>
    </tr>
    <tr>
      <th>CAGR</th>
      <td>7.36%	</td>
      <td>3.65%	</td>
      <td>4.88%</td>
    </tr>
    <tr>
      <th>MDD</th>
      <td>-33.73%	</td>
      <td>-29.06%	</td>
      <td>-21.64%</td>
    </tr>
    <tr>
      <th>SharpRatio</th>
      <td>0.19</td>
      <td>0.13</td>
      <td>0.16</td>
    </tr>      
    <tr>
      <th>VaR</th>
      <td>-1.12</td>
      <td>-0.49</td>
      <td>-0.68</td>
    </tr>
    <tr>
      <th>CVaR</th>
      <td>-1.90</td>
      <td>-0.98</td>
      <td>-1.10</td>
    </tr>        
  </tbody>
</table>
</div>
<br>
<br/>

# __결론__

## __한계점__

  + Risk Parity 모델 구축에 있어서, 최소화할 때 특정 경우에 따라 자산의 RC가 동일하게 나오지 않을 상황이 생길 수 있다. 이 때 다른 방식으로 최소화 결과값을 구할 경우와 차이가 발생할 수 있다.
  
    실제로 Bridgewater사에서 All Weather Portfolio를 활용할 때 leverage를 적극 활용한다. 프로젝트 간 구성한 모델은 leverage가 활용되지 않아서 60/40 모델보다 저조한 수익률을 나타내고 있다. 또한 ETF를 활용할 때 지출되는 수수료, rebalancing 때 발생하는 거래수수료, 채권에서 나오는 coupon, 주식에서 나오는 배당 등 주가 외에 추가적으로 발생하는 소득과 지출에 대한 내용을 적용하지 않아 실제로 포트폴리오를 구성했을 때와 오차가 발생할 것이다.

## __결론__
 + 세계적인 대공황을 잘 방어하는 포트폴리오를 구성하기 위하여All Weather Portfolio를 구성하였다. 60/40 Portfolio 모델과 GMV Portfolio 모델과 비교해 보았을 때 수익률과 Sharpe Ratio는 두 모델의 중간의 성격을 띄는 모습을 보였다. 하지만 MDD는 세 모델 중 가장 낮은 모습을 보여주고 있다. MDD는 하락세에서 얼마나 방어할 수 있는지 보여주는 지표이기 때문에 목적에 잘 맞는 포트폴리오를 구성했다고 말할 수 있다.

