# -*- coding: utf-8 -*-

import pandas as pd
pd.options.display.max_columns=100
from sklearn.datasets import load_breast_cancer

data=load_breast_cancer()
X=pd.DataFrame(data.data, columns=data.feature_names)
y=pd.Series(data.target)

X.head()
y.head()
y.value_counts()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=1)



# 앙상블 클래스의 로딩
# - 부스팅 계열
# - 앙상블을 구현하는 내부의 각 모델들이 선형으로 연결되어 학습 및
#   예측을 수행하는 방법론

# 1. AdaBoost : 데이터 중심의 부스팅 방법론을 구현
#               (직전 모델이 잘못 예측한 데이터에 가중치를 부여하는 방법) 


# 2. GradientBoosting : 오차에 중심을 둔 부스팅 방법론을 구현
#                       (각각의 학습 데이터에 대해서 오차의 크기가 큰 데이터에
#                        가중치를 부여하여 전체 오차를 줄여나가는 방법)

# 부스팅 계열의 데이터 예측 예시
# 1번째 모델의 예측값 * 가중치(1번째 모델의 가중치는 작음) + 
# 2번째 모델의 예측값 * 가중치(번째 모델의 가중치) + 
# ...
# n번째 모델의 예측값 * 가중치(n번째 모델의 가중치는 큼)

# AdaBoost 클래스는 부스팅을 구현하기 위한 기본 모델일 직접 설정해야함
from sklearn.ensemble import AdaBoostClassifier

# 앙상블을 구현하기 위한 기본 클래스 로딩
from sklearn.linear_model import LogisticRegression

# 부스팅 계열의 베이스 모델은 강한 제약을 설정하여 테스트함
# (C의 값이 작으면 강한 제약)
base_estimator = LogisticRegression(C = 0.001,
                                    class_weight='balanced',
                                    n_jobs=-1,
                                    random_state=1)


model = AdaBoostClassifier(base_estimator=base_estimator,
                           learning_rate=1.0,
                           random_state=1,
                           n_estimators=150)

model.fit(X_train, y_train)
score = model.score(X_train, y_train)
print(f'score (Train) : {score}')

score = model.score(X_test, y_test)
print(f'score (Test) : {score}')

# 테스트를 높이는 법 sample,feature의 수를 최대치인 1.0으로 주던가
# 반대로 낮추는 법은 아예 DesicionTree의 depth를 제한해주면 됨
# 배깅을 사용한 결정트리가 어쨌든 voting이 일어나기 때문에 다수결로 경계에 있는
# 샘플들은 분류 선이 심플하게 만들어진다~!




































