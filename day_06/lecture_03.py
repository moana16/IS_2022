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
# - 랜덤포레스트 : 배깅 방법론에 결정 트리를 조합하여 사용하는
#                 패턴이 빈번하게 발생하여 해당 구조를 하나의
#                 앙상블 모형으로 구현해 높은 클래스(가장마니 사용됨)
# - 배깅 + 결정트리
from sklearn.ensemble import RandomForestClassifier

# 앙상블을 구현하기 위한 내부 모델의 클래스 로딩

model = RandomForestClassifier(n_estimators=50,
                               max_depth=None,
                               max_samples=0.3,
                               max_features='auto',
                               n_jobs=-1,
                               random_state=1)
model.fit(X_train, y_train)
score = model.score(X_train, y_train)
print(f'score (Train) : {score}')

score = model.score(X_test, y_test)
print(f'score (Test) : {score}')

# 테스트를 높이는 법 sample,feature의 수를 최대치인 1.0으로 주던가
# 반대로 낮추는 법은 아예 DesicionTree의 depth를 제한해주면 됨
# 배깅을 사용한 결정트리가 어쨌든 voting이 일어나기 때문에 다수결로 경계에 있는
# 샘플들은 분류 선이 심플하게 만들어진다~!




































