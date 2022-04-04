# -*- coding: utf-8 -*-

# 트리모델 (분류)

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




from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3,
                               class_weight=None,
                               random_state=1)
# 기본값으로 "gini" 지수를 사용하여 분류를함
# max_Depth를 사용하여 과대적합이 나오지 않도록 지정해줌
# min_samples_split 숫자가 너무 크면 너무 뭉뚱그려서 분류하게됨

model.fit(X_train, y_train)

model.score(X_train, y_train) # 맟출때까지 depth를 내려감 - > 과대적합
model.score(X_test, y_test)

# 특정(컬럼)의 중요도 값 확인
print(f'feature_importances_ :  {model.feature_importances_}')









































