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




# GradientBoost 클래스는 부스팅을 구현하기 위한 기본 모델이 결정트리로 고정
# - 랜덤포레스트의 부스팅 버전 !
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
                           learning_rate=0.1,
                           max_depth=1,
                           subsample=0.3,
                           max_features=0.3,
                           random_state=1,
                           n_estimators=200,
                           verbose=3)

model.fit(X_train, y_train)
score = model.score(X_train, y_train)
print(f'score (Train) : {score}')

score = model.score(X_test, y_test)
print(f'score (Test) : {score}')

# 테스트를 높이는 법 sample,feature의 수를 최대치인 1.0으로 주던가
# 반대로 낮추는 법은 아예 DesicionTree의 depth를 제한해주면 됨
# 배깅을 사용한 결정트리가 어쨌든 voting이 일어나기 때문에 다수결로 경계에 있는
# 샘플들은 분류 선이 심플하게 만들어진다~!




































