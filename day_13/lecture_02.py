# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs
X, y =make_blobs(n_samples=150,
                 n_features=2,
                 centers=3,
                 cluster_std=0.5,
                 shuffle=True,
                 random_state=0)

X[:10]
y[:10] #centers=3으로 지정해두었기 때문에 정답데이터가 세개임..

import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],c='white',
            marker='o', edgecolors='black',
            s=50)
plt.grid()
plt.show()

# 최근접 이웃 알고리즘
# 군집 분석을 위한 클래스
# kMeans
# - 가장 많이 사용되는 군집 분석 클래스
# - 알고리즘이 단순하고 변경에 용이
#  (수정사항의 반영이 손쉬움)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=5,# 몇개를 이용해서 군집 분석 할건지
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0) 
# 군집 분석 알고리즘인 KMeans의
# fit 메소드의 동작
# - n_clusters에 정의된 개수만큼 포인트를 지정하여
#   최적의 위치를 찾도록 검색하는 과정을 수행

km.fit(X)

# 군집의  결과를 생성하여 반환
y_cluster = km.predict(X)
print(y_cluster)

plt.scatter(X[y_cluster==0, 0],
            X[y_cluster==0, 1],
            s=50,
            c='lightgreen',
            marker='s',
            label = 'Cluster 1')

plt.scatter(X[y_cluster==1, 0],
            X[y_cluster==1, 1],
            s=50,
            c='orange',
            marker='o',
            label = 'Cluster 2')

plt.scatter(X[y_cluster==2, 0],
            X[y_cluster==2, 1],
            s=50,
            c='lightblue',
            marker='v',
            label = 'Cluster 3')

plt.scatter(X[y_cluster==3, 0],
            X[y_cluster==3, 1],
            s=50,
            c='yellow',
            marker='v',
            label = 'Cluster 3')

plt.scatter(X[y_cluster==4, 0],
            X[y_cluster==4, 1],
            s=50,
            c='blue',
            marker='v',
            label = 'Cluster 3')

plt.scatter(km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],
            s=250,
            c='red',
            marker='*',
            label = 'Center')


plt.legend()
plt.grid()
plt.show()




















