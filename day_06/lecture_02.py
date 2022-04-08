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
# - 배깅 : 특정 머신러닝 알고리즘 기반으로 데이터의 무작위 추출을 사용하여
#          각 모델들이 서로다른 데이터릃 학습하는 방식으로
#          앙상블을 구현하는 방법

from sklearn.ensemble import BaggingClassifier

# 앙상블을 구현하기 위한 내부 모델의 클래스 로딩

from sklearn.tree import DecisionTreeClassifier


base_estimator = DecisionTreeClassifier(random_state=1,
                                        max_depth=3)



model = BaggingClassifier(base_estimator=base_estimator,
                         n_estimators=50,
                         max_samples=0.3,
                         max_features=0.3,
                         random_state=1,
                         n_jobs=-1)

model.fit(X_train, y_train)
score = model.score(X_train, y_train)
print(f'score (Train) : {score}')

score = model.score(X_test, y_test)
print(f'score (Test) : {score}')

# 학습을 높이는 법 sample,feature의 수를 최대치인 1.0으로 주던가
# 반대로 낮추는 법(테스트 높이는 법)은 아예 DesicionTree의 depth를 제한해주면 됨 혹은 위에 값 조절
# 배깅을 사용한 결정트리가 어쨌든 voting이 일어나기 때문에 다수결로 경계에 있는
# 샘플들은 분류 선이 심플하게 만들어진다~!
# 결정트리 사용하는 이유? 전처리 노력 별로 안필요하고 과적합하기 좋음
# 일반적으로 성능이 보장되어서 나옴




































