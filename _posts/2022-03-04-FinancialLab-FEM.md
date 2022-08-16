---
category: [FinancialLab][2020-2]
title : "Panel Data Analysis"
excerpt : ""

date: 2022-03-01
use_math : true
mathjax : true
---

#  __국내 인적자본을 고려한 생애주기별 자산관리__

+ 인적자본을 분석하기 위해 고정효과 회귀분석을 통해 인적자본 모델 추정
+ 노동임금을 추정하는 알고리즘은 나이 더미변수, 혼인여부, 가족 구성원수로 구성
$
y_{it} = \beta_0 + \beta_1X_{it} + \beta_2Z_i+ \epsilon_{it} 
$
+ $X_{it}$(벡터) : 나이 더미변수
+ $Z_{t}$(벡터) : 고정효과 변수

# __패널 데이터 분석__

+ 패널데이터 : 종단면 데이터 or 횡단면 시계열 데이터, 고정된 entity(개인,  기업 등)에서 시간 경과에 따른 관측치로부터 파생된 데이터

    ex) n명의 고정된 개인을 대상으로 소득, 학력, 직업, 노동시간을 10년간 모은 데이터 


+ 패널데이터 분석의 종류
1) Fixed Effect Model
2) Random Effect Model 

$
y_{it} = \beta_0 + \beta_1x_{it} + u_i+ \epsilon_{it} 
$
+ $u_i + \epsilon_{it}$ : $u_i$는 개별 특성에 의한 오차, $\epsilon_{it}$ 는 시간변수에 따른 오차
+ $u_i$는 내생성(독립변수와 오차항의 covariance 존재) 문제 야기

## FEM과 REM 차이

+ 잔차 제곱합
$
Q = \sum \sum (y_{it} - \hat\beta x_{it})^2 = \sum\sum[(w_1y_{it}- w_2\bar y_i)- \hat\beta(w_1x_{it}-w_2\bar x_i)]^2\\
=w_1^2\sum\sum[(y_{it}-\bar y_i) - \hat\beta(x_{it}-\bar x_i)]^2  + (w_1 - w_2)^2 \sum\sum (\bar y_i - \hat\beta \bar x_i)^2
$

+ Within Variation : $ w_1^2\sum\sum[(y_{it}-\bar y_i) - \hat\beta(x_{it}-\bar x_i)]^2$

    : 한 entity 내에서 발생하는 변동

+ Between Variation : $ (w_1 - w_2)^2 \sum\sum (\bar y_i - \hat\beta \bar x_i)^2 $ 

    : entity 사이의 변동
    
+ $ w_1 = w_2 = 1 $ 일때 between variation = 0, FEM(Between Variation 통제) 

# __고정효과 모형(FEM)__

+ $u_i + \epsilon_{it}$ 에서 $u_i$는 내생성 문제를 야기하기 때문에 제거하여 풀이 (내생성 문제는 OLS 추정치가 편향되게 만듦)

+ 내생성이 생성되는 3가지 가능성

    + Simultaneity : 시스템의 동시 작동성 (X가 Y에 영향을 줌, Y도 X에 영향을 줌) 

        -> 두개의 회귀분석 후 두 개의  연립방정식 계산
    + Measurement Error 
    + Omitted Variable : 설명변수의 누락
    (Omitted Variable bias 는 𝑢_𝑖와 관련하여 문제가 있다. Ex) 임금 = 경력, 학력, 노동시간, 거주지역 으로 표현하자고 할때, “능력＂이라는 요소를 포함X -> 능력은 𝑢_𝑖에 포함되게 된다. 능력은 학력과 경력에 양의 상관관계를 가질 것이라는 것 예측 가능)


# __고정효과 모형의 활용__

+ Durbin-Wu-Hausman Test를 수행하고 Test가 기각 될때 FEM 사용이 바람직함
 $
H = (b_1 - b_0 )'(Var(b_0)-Var(b_1))^t(b_1- b_0)
 $

<br></br>
 # __고정효과 모형의 장단점__

+ 장점 : 관찰할 수 없지만 시간에 따라 변하지 않는 Entity의 고유 특성 통제

+ 단점 : FEM으로 시간에 따라 변하지 않는 변수가 종속변수에 미치는 영향을 분석할 수 없음
  
    : Measurement Error가 존재하는 경우 내생성의 문제 
    
    : 자유도의 상실이 커 추정량의 효율성이 낮아짐


# __Random Effect Model__

+ FEM : $w_1 = w_2 = 1,\; u_i + \epsilon_{it}$

    : Within variation만 사용

+ REM : 
$
w_1 = 1, \; w_2 = 1 - \sqrt{\frac{1+\hat p}{1+(t-1)\hat p}}, \; \hat p = \frac{\hat \sigma_u^2}{\hat \sigma_u^2 + \hat \sigma_\epsilon^2} 
$
$
\hat \sigma_v^2 = cov(V_{it}, V_{is}), \; \hat \sigma_u^2 + \hat \sigma_\epsilon^2 = var(V_{it})
 $

 + Random effect는 Within Variation과 Betwwen variation 중 편차가 더 작은 값에 가중치를 준다. (퀄리티가 더 좋은 정보를 사용)

 + REM의 가중치는 GLS 추정을 통해 도출한다.

 ## __Genealized Least Squares(GLS)__

+ 오차항에 이분산 or 자기상관의 문제가 있을 때 OLS는 불편추정량(Unbiased), 효율적인 추정량 X
이 때 GLS가 가장 효율적인 추정량(BLUE)

+ 이분산 Heteroskedasticity: 오차항의 분산이 회귀모형의 포함되는 변수에 따라 일정하게 나타나지 않는 현상

<p>
<img src = "/assets/img/FEM_GLS.png"  >
</p>

이때 $ \Omega=PP' $을 만족하는 P가 존재한다고 하자 이 때 $P^{-1}$을 L로 두면 $L'L = \Omega$ 를 만족하는 비특이행렬 L이 존재한다. 

회귀모형 $Y = XB + U, E(UU') = \sigma^2\Omega$에 L을 곱하여 변환

$$ E(LU) = L*E(U) = 0\\
E(LU(LU)') =E(LUU'L')= L\sigma^2\Omega L' = \sigma^2 LL^{-1} L'^{-1}L' = \sigma^2I
 $$

 변환된 모형의 에러는 평균이 0이고 분산이 자기상관 되어있지 않다.

$$
\hat \sigma_u^2 = \frac{e'\Omega^{-1}e}{n-k}\\
Var(\hat{B_G}) = \hat \sigma_u^2(X' \Omega^{-1}X)^{-1}
$$


+ GLS는 자기상관성을 상쇄할 수 있도록 가중치를 부여하여 최소제곱 추정을 하는 것이다.
Random Effect Model에서도 같은 방식으로 가중치를 부여한다.

# __패널데이터 검정__

## __Hausman Test__

+ 서로 다른 추정기 2개를 비교한 계량모형의 오차검정

$
H_0 : cov(x_{it}, u_i) = 0 \\
H_1 : cov(x_{it}, u_i) \neq 0 
$

+ $cov(x_{it}, u_i) = 0$ 이라는 가정이 성립하면 FEM, REM 모두 일치추정량으로 유사함

+ 귀무가설이 맞다면 FEM과 REM 모두 일치 추정량으로 차이가 존재하지 않음 → REM or FEM 둘 다 사용 가능

+ 귀무가설이 틀리다면 REM은 일치 추정량이 아니므로 고정효과 모형을 선택 $𝑐𝑜𝑣(𝑥_{𝑖𝑡},𝑢_𝑖 )=0$ 

## __F-검정__

+ 오차항의 고정된 개체특성을 고려할 필요가 있는지 확인
+ $H_0$ : 오차항 = 0
+ 기각되지 않으면 Pooled OLS, 기각 될 경우 Fixed Effect Model

## __Breusch Test(Xttest0)__

+ 모델의 이분산성을 검사하는 모형
+ Pooled OLS와 Random Effect Model 중 어떤 모형이 적절한지
+ 귀무가설 : Pooled OLS가  적절함

## __Xttest1(stat)__

+ Breasch Test의 확장형, 확률효과가 존재하는지 결과와 더불어 e_it의 자기상관 검증


## __Xttest2__

+ 동시적 상관관계 검정
+ 귀무가설 : 패널 개체간 동시적 상관이 존재하지 않는다.


# __Code of FEM__ 

+ R을 활용하여 코드를 작성하였습니다.

```R
library(plm)
library(readr)
library(tidyverse) 
library(car)       
library(gplots)    
library(tseries)   
library(lmtest)   


dataPanel101 <- read_csv("https://github.com/ds777/sample-datasets/blob/master/dataPanel101.csv?raw=true")
dataPanel101 <- plm.data(dataPanel101, index=c("country","year"))

#OLS
ols <- lm(y~ x1, data = dataPanel101)
summary(ols)

yhat <- ols$fitted


ggplot(dataPanel101, aes(x=x1, y=y)) + geom_point() + geom_smooth(method=lm)


#Fixed effect model by Country
fixed.dum <-lm(y~x1 + factor(country) -1, data=dataPanel101)

summary(fixed.dum)

yhat2 <- fixed.dum$fitted
scatterplot(yhat2 ~ dataPanel101$x1 | dataPanel101$country,  xlab ="x1", ylab ="yhat", boxplots = FALSE,smooth = FALSE)
abline(lm(dataPanel101$y~dataPanel101$x1),lwd=3, col="red")

#Fixed effect model
fixed <- plm(y~ x1, data = dataPanel101, model="within")
summary(fixed)

#Fixed vs OLS
pFtest(fixed, ols)

# Display the fixed effects (constants for each country)
fixef(fixed)

#Random Effect Model
random <- plm(y ~ x1, data=dataPanel101, model="random")
summary(random)

#Hausman test
phtest(fixed, random)



#FEM by time
fixed.time <- plm(y ~ x1 + factor(year), data=dataPanel101, model="within")
summary(fixed.time)


# Testing time-fixed effects. The null is that no time-fixed effects are needed
pFtest(fixed.time, fixed)

plmtest(fixed, c("time"), type=("bp"))

#Pooled OLS

pool <- plm(y ~ x1, data=dataPanel101, model="pooling")
summary(pool)

# Breusch-Pagan Lagrange Multiplier for random effects. Null is no panel effect (i.e. OLS better).
plmtest(pool, type=c("bp"))


# Heteroskedasticity testing 
bptest(y ~ x1 + factor(country), data = dataPanel101, studentize=F)

```
