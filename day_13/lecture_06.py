# -*- coding: utf-8 -*-
# 병합군집
# - 다수개의 소규모 군집을 랜덤하게 생성
# - 다수개의 소규모 군집을 취합해 하나로 병합 (인접한 위치의 군집사이에서 발생)
# - 원하는 개수의 군집으로 최종 처리를 완료
#이건 육번임
from sklearn.datasets import make_moons
X,y =make_moons(n_samples=200,
                noise=0.05,
                random_state=0)

X[:10]
print(y[:10])

import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1])
plt.show()


# KMeans 클래스
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)
km.fit(X)

y_cluster=km.predict(X)

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


plt.scatter(km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],
            s=250,
            c='pink',
            marker='*',
            label = 'Center')


plt.legend()
plt.grid()
plt.show()

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2)
            
ac.fit(X)

y_cluster=ac.fit_predict(X)

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


plt.legend()
plt.grid()
plt.show()