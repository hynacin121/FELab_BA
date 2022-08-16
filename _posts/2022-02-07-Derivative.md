---

category: [FinancialEngineering]
title : "Derivative"
excerpt : 
date: 2022-02-07
use_math : true
---


# __Derivative Security__

+ a security whose payoff is explicitly tied to the value of some other variable
+ __Underlying Security__ : security that determines the value of derivatibe security

+ Main type of derivative security : __Forward Contracts__, __Future Contracts__, __Options__, __Swaps__

# __Forward Contracts__

+ A contract to purchase or sell a specific amount of the underlying asset at a specific price and time in the future

     ex) Purchase 100,000 pounds of sugar at 12 cents per pound on the 15th of March next year
+ Buyer is said to be long, seller is said to be short.

+ Forward contracts can eliminate the risk associted with the price of commodities.
+ Both suppliers and consumers of large quantities of a commodity frequently find it advantageous to lock in the price adsociated with future delivery

+ Specify that claims are settled at the defined future date

# __Forward Prices__

+ There are two prices associated with a forward contract : forward price and current value

+ Forward price F is the delivery price of a unit of the asset to be delivered at a specific future date
+ Current value of a forward contract is denoted by f

## __Forward Price formula__

+ Suppose an asset can be stored at zero cost and also sold short. Suppose the current spot price of the asset is S. The theoretical forward price F is
$
F = \frac S{d(0,T)}
$
where $d(0,T)$ is the dicount factor between 0 and T.

+ Costs of carry : holding a physical asset entails storage costs. These costs affect the theoretical forward price.

$ F = \frac S{d(0,M)} + \sum^{M-1}_{k=0}\frac{c(k)}{d(k,m)}
$
where $d(k,m)$ is discount factor from k to M, c(k) is holding cost per unit in period k.


# __Swaps__

+ an agreement to exchange one cash flow steam for another

    ex) Consider an interest rate swap
    
    Party A agrees to make a series of semiannual payments to party B equak to fixed rate of interest on a notional principal. Party B makes a series of semiannual payments to party A based on a floating rate of interest.

## __Value of a Commodity Swap__

+ We can value this using the concepts of forward markets

$
V = \sum^M_{i=1}d(0,i)(F_i -X)N
$
where $d(0,i)$ is the discount factor at time 0 for cash received at i, 
$F_i$ is forward price of one unit of the commodity to be received at i.

# __Option__

+ __Option__ : Option is the right, but not the obligation, to buy or sell an asset under specified terms.
+ Call option : right to purchase something
+ Put option : right to sell something
+ Usually Option has a price; we refer to this price as the option premium

## __Call Option__

+ right to buy an asset in the future
+ Want to buy the asset with lower risk
+ Want to invest in an asset without owning the asset

# __Option Concepts__

## __3 Components__
1. Description of what can be bought or sold
2. the excercise price must be specified
3. period of time for which the option is valid must be specified

+ American option : allows exercise at any time before the expiration
+ European Option : allows exercise only on the expiration date

+ There are two sides to any option
    + grants the option is said to write an option (short position)
    + obtaions the option is said to purchase it (long position)

+ The party purchasing an option faces no risk of loss other than the original purchase premium.

# __Option Values__

+ Final goal is to determine the current value of an option
+ K : strike price of K
+ S : on the expiration date, the price of the underlyung stock

## __Call Option Values__

+ If $ S < K $, then option value is zero
+ If $ S > K $, then option does have value
    + Your profit would be $ S - K $, which is therefore the value of the option.
+ Thus, the value of the call option at expiration is 
$
C = max(0, S-K)
$

## __Put Option Values__

+ If $ S > K $, then option value is zero
+ If $ S < K $, then option does have value, profit would be $ K - S $

## __Time Value of Options__

+ Option focused on the value of  an option at expiration
+ When there is longer time to expiration, the value of an option is higher (long position)

## __Volatility of the underlying stock__
+ Value of an option increases with higher volatility.

# __Option Combinations and Put-Call Parity__

+ Combination of options in order to implement special hedging and speculative strategies.
+ 그래프 넣기 P50


# Put-Call Parity

+ Buy one call and sell one put with same strike -> a straght line with slope = 1.
+ Recieve an amount K 

-> a line originating at the origin with slope = 1

+ Let C and P be the prices of a European call and a European put, both with a strike price of K and both  defined on the same stock with price S.
$
C - P + dK = S
$
where d is the discount factor to the expiration date.

$
Call + Cash = Put + Underlying stock
$





=======
---

category: [FinancialEngineering]
title : "Derivative"
excerpt : 
date: 2022-02-07
use_math : true
---


# __Derivative Security__

+ a security whose payoff is explicitly tied to the value of some other variable
+ __Underlying Security__ : security that determines the value of derivatibe security

+ Main type of derivative security : __Forward Contracts__, __Future Contracts__, __Options__, __Swaps__

# __Forward Contracts__

+ A contract to purchase or sell a specific amount of the underlying asset at a specific price and time in the future

     ex) Purchase 100,000 pounds of sugar at 12 cents per pound on the 15th of March next year
+ Buyer is said to be long, seller is said to be short.

+ Forward contracts can eliminate the risk associted with the price of commodities.
+ Both suppliers and consumers of large quantities of a commodity frequently find it advantageous to lock in the price adsociated with future delivery

+ Specify that claims are settled at the defined future date

# __Forward Prices__

+ There are two prices associated with a forward contract : forward price and current value

+ Forward price F is the delivery price of a unit of the asset to be delivered at a specific future date
+ Current value of a forward contract is denoted by f

## __Forward Price formula__

+ Suppose an asset can be stored at zero cost and also sold short. Suppose the current spot price of the asset is S. The theoretical forward price F is
$
F = \frac S{d(0,T)}
$
where $d(0,T)$ is the dicount factor between 0 and T.

+ Costs of carry : holding a physical asset entails storage costs. These costs affect the theoretical forward price.

$ F = \frac S{d(0,M)} + \sum^{M-1}_{k=0}\frac{c(k)}{d(k,m)}
$
where $d(k,m)$ is discount factor from k to M, c(k) is holding cost per unit in period k.


# __Swaps__

+ an agreement to exchange one cash flow steam for another

    ex) Consider an interest rate swap
    
    Party A agrees to make a series of semiannual payments to party B equak to fixed rate of interest on a notional principal. Party B makes a series of semiannual payments to party A based on a floating rate of interest.

## __Value of a Commodity Swap__

+ We can value this using the concepts of forward markets

$
V = \sum^M_{i=1}d(0,i)(F_i -X)N
$
where $d(0,i)$ is the discount factor at time 0 for cash received at i, 
$F_i$ is forward price of one unit of the commodity to be received at i.

# __Option__

+ __Option__ : Option is the right, but not the obligation, to buy or sell an asset under specified terms.
+ Call option : right to purchase something
+ Put option : right to sell something
+ Usually Option has a price; we refer to this price as the option premium

## __Call Option__

+ right to buy an asset in the future
+ Want to buy the asset with lower risk
+ Want to invest in an asset without owning the asset

# __Option Concepts__

## __3 Components__
1. Description of what can be bought or sold
2. the excercise price must be specified
3. period of time for which the option is valid must be specified

+ American option : allows exercise at any time before the expiration
+ European Option : allows exercise only on the expiration date

+ There are two sides to any option
    + grants the option is said to write an option (short position)
    + obtaions the option is said to purchase it (long position)

+ The party purchasing an option faces no risk of loss other than the original purchase premium.

# __Option Values__

+ Final goal is to determine the current value of an option
+ K : strike price of K
+ S : on the expiration date, the price of the underlyung stock

## __Call Option Values__

+ If $ S < K $, then option value is zero
+ If $ S > K $, then option does have value
    + Your profit would be $ S - K $, which is therefore the value of the option.
+ Thus, the value of the call option at expiration is 
$
C = max(0, S-K)
$

## __Put Option Values__

+ If $ S > K $, then option value is zero
+ If $ S < K $, then option does have value, profit would be $ K - S $

## __Time Value of Options__

+ Option focused on the value of  an option at expiration
+ When there is longer time to expiration, the value of an option is higher (long position)

## __Volatility of the underlying stock__
+ Value of an option increases with higher volatility.

# __Option Combinations and Put-Call Parity__

+ Combination of options in order to implement special hedging and speculative strategies.
+ 그래프 넣기 P50


# Put-Call Parity

+ Buy one call and sell one put with same strike -> a straght line with slope = 1.
+ Recieve an amount K 

-> a line originating at the origin with slope = 1

+ Let C and P be the prices of a European call and a European put, both with a strike price of K and both  defined on the same stock with price S.
$
C - P + dK = S
$
where d is the discount factor to the expiration date.

$
Call + Cash = Put + Underlying stock
$



