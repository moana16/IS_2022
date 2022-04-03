# -*- coding: utf-8 -*-

# 선형모델 (분류)

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




from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2',
                           C=1.0,
                           #class_weight='balanced',
                           class_weight={0:1000,1:1}, # 0에대가 가중치 1000, 1에다가 가중치 1
                           solver='lbfgs',
                           max_iter=1000000,
                           n_jobs=-1,
                           random_state=5,
                           verbose=1)
# l1 = rasso, l2 = lidge 모든 특성에 대한 가중치를 적절하게 나눠줌? 
# 알파값이 크면 제약이 큼 - > 0주변에 밀집됨
# 근데 여기서는 C(제약 크기 조절 변수)는 크면 제약이 작음 - > 가중치가 널띔
# class_weight : default는 데이터값이 단순히 큰 쪽을 선호
#                0의 개수가 적으니까 1과 밸런스를 맞춰봐 = 'balanced'

model.fit(X_train, y_train)

model.score(X_train, y_train)
model.score(X_test, y_test)

# 가중치 값 확인
print(f'coef_ :  {model.coef_}')

# 절편 값 확인
print(f'intercept_ : {model.intercept_}')


proba = model.predict_proba(X_train[:5])
proba

pred = model.predict(X_train[:5])
pred

df = model.decision_function(X_train[:5])
df

y_train[:5]


# 분류 모델의 평가 방법
# 1. 정확도
#  - 전체 데이터에서 정답으로 맞춘 비율
#  - 머신러닝 모델의 score 메소드 
#  - 분류하고자 하는 각각의 클래스의 비율이 동일한 경우에만 사용

# 2. 정밀도
#  - 집합 : 머신러닝 모델이 예측한 결과
#  - 위의 집합에서 각각의 클래스 별 정답 비율
# (머신러닝이 예측한 날에 상한가를 친다 - > 정밀도가 높음)


# 3. 재현율
#  - 집합 : 학습 데이터 셋
#  - 위의 집합에서 머신러닝 모델이 예측한 정답 비율
# (상한가 치는날은 무조건 머신러닝이 예측함 - > 재현율이 높음)

# 혼동행렬
from sklearn.metrics import confusion_matrix

pred = model.predict(X_train)
cm = confusion_matrix(y_train, pred)
cm


y_train.value_counts()

# 머신러닝 모델의 예측    0     1
# 실제 0인 데이터      [[141,   7]   = 148
# 실제 1인 데이터       [  5, 245]]  = 250

# 정밀도(0) : 141 / (141+ 5)
# 재현율(0) : 141 / (141 + 7)

# 정확도
from sklearn.metrics import accuracy_score
# 정밀도
from sklearn.metrics import precision_score
# 재현율
from sklearn.metrics import recall_score

pred = model.predict(X_train)
# 0에 대한 정밀도 값
ps = precision_score(y_train, pred , pos_label=0)
ps
# 0에 대한 재현율 값
rs = recall_score(y_train, pred , pos_label=0)
rs










































