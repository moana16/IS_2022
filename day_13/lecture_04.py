# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')
X.head()
X.info()
X.describe()

# 머신러닝 모델의 학습 결과
# 가중치가 할당되지 않은 혹은 가중치가 굉장히 작은
# 컬럼들에 대해서 새로운 특성을 생성(유의미하게 만들어주겠다는)
# - 차원축소
# - 군집분석

X_part = X[['radius error',
           'compactness error',
           'concavity error']]


from sklearn.cluster import KMeans

# 최적의 군집(클러스터)의 개수를 검색하는 방법
# - 엘로우 방법을 활용하여 처리할 수 있음
values = []
for i in range(1,15) :
    km = KMeans(n_clusters=i,# 몇개를 이용해서 군집 분석 할건지
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0) 
    km.fit(X_part)
    # 클러스터 내의 각 클래스의 SSE 값을 반환하는 
    #inertia_ 속성 값
    values.append(km.inertia_)
print(values)

import matplotlib.pyplot as plt

plt.plot(range(1,15), values, marker='o')
plt.xlabel('number of cluster')
plt.ylabel('inertia_')
plt.show()

# - 5개의 군집 개수가 최적화

km = KMeans(n_clusters=5,# 몇개를 이용해서 군집 분석 할건지
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0) 
km.fit(X_part)

X['cluster_result']=km.predict(X_part)
del X['radius error']
del X['compactness error']
del X['concavity error']

print(X.info())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=.3,
    stratify=y,
    random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0,
                           class_weight='balanced',
                           random_state=0)
model.fit(X_train_scaled, y_train)
v_score = model.score(X_train_scaled, y_train)
print(f'학습 :  {v_score}')
v_score = model.score(X_test_scaled, y_test)
print(f'테스트 :  {v_score}')

print(f'학습된 가중치 : \n{model.coef_}')







