---
category: [FinancialLab][2020-2]
title : "Clustering"
excerpt : ""

date: 2022-03-10
use_math : true
mathjax : true
---



# __Clustering__

+ 정의 : 클러스터링이란 유사한 성격을 가진 개체를 묶어 그룹으로 구성하는 것
+ 비지도 학습
+ 종류 : Hard Clustering, Soft Clustering, Partitional Clustering, Hierarchical Clustering, Self-Organizing Map, Spectual Clustering.



## __Hard Clustering__

+ __Hard Clustering__ : Grouping Data items such taht each item is only assigned to one cluster.

### __K-Means Clustering__
+ Unsupervised clustering algorithm that is used to group into k-clustering

+ Repeat 2 steps below until clusters and their mean is stable:
    1. For each data item, assign it to the nearest cluster center. Nearest distance can be calculated based on distance algorithms
    2. Calculate mean of the cluster with all data items

+ Algorithm 
$
X = C_1 \cup \dots \cup C_k, C_i \cap C_j = \varnothing \\
\argmin_c \sum_{i=1}^K\sum_{x_j \in C_i} \left\vert\right\vert x_j - c_i \left\vert\right\vert^2
$


+ Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



```


```python
Kmean = KMeans(n_clusters=2)
Kmean.fit(a)
Kmean.cluster_centers_
```




    array([[-1.74931843, -1.49048386],
           [ 3.53161236,  3.57139274]])




```python
a = -3 * np.random.rand(100,2)
b = 2 + 3 * np.random.rand(50,2)
a[50:100, :] = b
plt.scatter(a[ : , 0], a[ :, 1], s = 50, c = 'b')
plt.scatter(-1.74931843, -1.49048386, s= 200 , c='g')
plt.scatter(3.53161236,  3.57139274, s= 200 , c='r')
plt.show()
```


    
<p>
<img src = "/assets/img/kmeans.png"  >
</p>
    


## __Soft Clustering__

+ __Soft Clustering__ : grouping the data items such that an item can exist in multiple clusters.

### __Fuzzy C-means Clustering__
+ Based on the fuzzy logic and is often referred to as the FCM algorithm.
+ Each point has a probability of belonging to each cluster, rather than completely belonging to just one cluster as it is the case in the traditional k-means.
+ Slower than K means. Each point is evaluated with each cluster, and more operations are involved in each evaluation.
+ Process flow of FCM
    1. Assume a fixed number of cluster _k_
    2. Initialization : Randomly initialize the k-means $\mu k$ associated with the clusters and compute the probability that each data point xi is a member of given cluster k
    3. Iteration : Recalculate the centroid of the cluster as the weighted centroid given the porbabilities of membership of all data points xi
    $
    \mu_k(n+1)= \frac{\sum_{x_i \in k}x_i*P(\mu_k \vert x_i)^b}{\sum_{x_i \in k}P(\mu_k \vert x_i)^b}
    $ 
    4. Termination : Iterate until convergence or until a user-specified number of iterations has been reached 

```python
import numpy as np

try:
  from fcmeans import FCM
except:
  %pip install pip fuzzy-c-means
  from fcmeans import FCM

from matplotlib import pyplot as plt
  
```

```python
a = -2 * np.random.rand(100,2)
b =  3 * np.random.rand(50,2)
a[50:100, :] = b
plt.scatter(a[ : , 0], a[ :, 1], s = 50, c = 'b')
plt.show()

```
   
    
<p>
<img src = "/assets/img/fcm_1.png"  >
</p>
 

```python
fcm = FCM(n_cluster = 2)
fcm.fit(a)

fcm_centers = fcm.centers
fcm_labels = fcm.predict(a)

f, axes = plt.subplots(1,2)
axes[0].scatter(a[:,0], a[:,1], alpha=.1)
axes[1].scatter(a[:,0], a[:,1], c=fcm_labels, alpha=.1)
axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=50, )

plt.show()
```
<p>
<img src = "/assets/img/fcm_2.png"  >
</p>

## __Hierarchical Clustering__
+ 계층적 군집 분석
+ Algorithm
    1. 가장 거리가 가까운 데이터를 찾아서 이를 묶는다.
    2. 1을 모든 데이터가 하나의 클러스터로 묶일때까지 반복 
    3. 클러스터를 나눈다. 

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```


```python
a = np.round(100 * np.random.rand(10,2), 1)

```


```python
labels = range (1, 11)
plt.figure(figsize = (7, 5))
plt.subplots_adjust(bottom=0.1)
plt.scatter(a[:,0],a[:,1])

for label,x, y in zip(labels, a[:,0],a[:,1]):
    plt.annotate(label, xy= (x, y), xytext = (-3,3), 
                   textcoords='offset points', ha='right', va='bottom')
    
plt.show()
```


<p>
<img src = "/assets/img/hie_1.png"  >
</p>



```python
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(a, 'single')
```


```python
labelList = range(1, 11)

plt.figure(figsize=(7, 5))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()
```

<p>
<img src = "/assets/img/hie_2.png"  >
</p>



## __Partitional Clustering__ 
+ Hierarchical Clustering과는 달리 cluster의 계층을 고려하지 않고 평면적으로 clustering하는 방법을 나타냄

