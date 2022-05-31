
# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs
X, y =make_blobs(n_samples=150,
                 n_features=2,
                 centers=3,
              
                 shuffle=True,
                 random_state=0)

X[:10]
y[:10]

import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],c='white',
            marker='o', edgecolors='black',
            s=50)
plt.grid()
plt.show()


from sklearn.cluster import KMeans

# 최적의 군집(클러스터)의 개수를 검색하는 방법
# - 엘로우 방법을 활용하여 처리할 수 있음
values = []
for i in range(1,11) :
    km = KMeans(n_clusters=i,# 몇개를 이용해서 군집 분석 할건지
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0) 
    km.fit(X)
    # 클러스터 내의 각 클래스의 SSE 값을 반환하는 
    #inertia_ 속성 값
    values.append(km.inertia_)
print(values)
       
plt.plot(range(1,11), values, marker='o')
plt.xlabel('number of cluster')
plt.ylabel('inertia_')
plt.show()



















