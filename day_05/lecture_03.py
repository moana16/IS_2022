# -*- coding: utf-8 -*-

# 서포트 벡터 머신 (분류)

import pandas as pd
pd.options.display.max_columns=100
from sklearn.datasets import load_breast_cancer

# 데이터 가져오기
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X.info()
X.isnull().sum()
X.describe(include='all') # 'all'을 넣으면 문자열에 해당하는 통계도 보여줌

y.head()
y.value_counts()
y.value_counts() / len(y) # 비율로 보기

from sklearn.model_selection import train_test_split
splits = train_test_split(X,y, test_size=0.3, random_state=10, stratify=y)

X_train=splits[0]
X_test=splits[1]
y_train=splits[2]
y_test=splits[-1]

X_train.head()
X_test.head()

X_train.shape
X_test.shape


y_train.value_counts() / len(y_train)
y_test.value_counts() / len(y_test)




from sklearn.svm import SVC, LinearSVC

model = SVC(C=1.0,
            gamma='scale',
            class_weight='balanced',
            random_state=1)

model.fit(X_train, y_train)

model.score(X_train, y_train) # 맟출때까지 depth를 내려감 - > 과대적합
model.score(X_test, y_test)

# 서포트 벡터의 가중치 값 확인
# - 커널 방법을 선형으로 (linear)로 설정한 경우 사용
print(f'coef_ :  {model.coef_}')
print(f'intercept_ :  {model.intercept_}')









































