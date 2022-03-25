# -*- coding: utf-8 -*-
# 회귀 분석
# - 머신러닝이 예측해야하는 정답의 데이터가 연속된 수치형인 경우를 의미함
# - 분류 분석의 경우 정답 데이터는 범주형[남자/여자, 탈퇴/유지 ...]
# - 선형 방정식을 활용한 머신러닝 실습

import pandas as pd
# load_ : 연습을 위한 간단한 데이터 셋
# fetch_ : 실 데이터 셋 (상대적으로 데이터의 개수가 많음)
from sklearn.datasets import fetch_california_housing


data = fetch_california_housing()

# 설명변수 : 집값 데이터를 예측하기 위한 피처 데이터를 저장하는 변수
#           (2차원 데이터)
X =pd.DataFrame(data.data, columns=data.feature_names)
# 종속변수(정답) : 설명변수를 사용하여 예측하기 위한 변수
#                 (설명변수의 행수와 동일한 크기의 1차원 배열)
y=pd.Series(data.target)

# 설명변수의 EDA (Non-null Count 확인, 변수타입 확인)
print(X.info())

# 결측 데이터의 개수를 확인하는 방법
print(X.isnull())
# 한눈에 파악
print(X.isnull().sum()) 

pd.options.display.max_columns=100
print(X.describe(include='all'))

# X 데이터를 구성하고 있는 각 특성(피처)들의
# 스케일 범위를 반드시 확인(Max, Min 확인해보긔)
# - Population 컬럼에서 스케일의 차이가 발생하는 것을 확인할 수 있음

# 스케일을 동일한 범위로 수정하기 위한 전처리 방법
# - 정규화, 일반화
# - StandardScaler, MinMaxScaler

# 각 컬러(특성, 피처)들에 대해서 산포도, 비율 등을 시각화 과정을
# 통해서 확인 및 전처리 체크를 수행

# 종속변수의 확인
# - 연속된 수치형 데이터 임을 확인
# - 회귀 분석을 위한 데이터 셋
print(y.head())
print(y.tail())

# 회귀 분석의 경우 중복되는 경우가 흔치 않으므로
# 분류 분석과 같이 value_counts 메소드를 사용하여
# 값의 개수를 확인하는 과정은 생략함
# print(y.value_counts())

# 데이터의 분할
# - 회귀분석을 위한 데이터 셋의  경우
#   y 데이터 내부의 값의 분포 비율을 유지할 필요가 없음
# - 비율이 중요한 경우에는 stratify를 사용하여 충화추출방법을 사용
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=1)

# shape 속성 : 데이터의 차원 정보를 반환
print(X_train.shape,X_test.shape)
# len 함수를 사용
print(len(y_train),len(y_test))

# 선형 방정식을 기반으로 회귀 예측을 수행할 수 있는 클래스
# - y = x1 * w1 + x2 * w2 + .... + xN * wN + b
# - LinearRegression 클래스의 학습은 x 데이터를 구성하는
#   각 컬럼(특성, 피처) 별 최적화된 가중치와 절편의 값을
#   계산하는 과정을 수행
from sklearn.linear_model import LinearRegression
# n_jobs : 사용가능한 코어 개수 사용 (-1은 모두 사용)

# 선형 모델에 제약조건 (L2, L1 제약조건)을 추가한 클래스
# L2 제약 조건 : 모든 특성에 대한 가중치의 값을
# 0 주변으로 위치하도록 제어하는 제약조건
# LinearRegression 클래스는 학습 데이터에 최적화되도록
#(오차가 최소화 되는 방향)
# 학습을 하기 때문에 테스트 데이터에 대한 일반화 성능이 감소
# 이러한 경우 모든 특성 데이터를 적절히 활용할 수 있도록 
# ???????????? /////////////////////////////////////////////////////추가해야함
# L2
# 모델의 일반화 성능( 테스트 데이터의 성능)
from sklearn.linear_model import Ridge,Lasso

# 머신러닝 객체의 생성
lr_model = LinearRegression(n_jobs=-1)

# Ridge, Lasso 클래스의 하이퍼 파라메테 alpha
# alpha 의 값이 커질수록 제약을 크게 설정
# (alpha의 값이 커질수록 모든 특성들의 가중치의 값은
# 0 주변으로 위치함)
# alpha 의 값이 작아질수록 제약이 약해짐
# (alpha 의 값이 작아질수록 모든 특성들의 가중치의 값은
# 0에서 멀어짐)
# alpha 의 값이 작아질수록 LinearRegression클래스와 동일해짐
ridge_model = Ridge(alpha=1.0, random_state=1)
lasso_model = Lasso(alpha=0.001, random_state=1)

# 학습
lr_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# Lasso 클래스를 사용하여 모델을 구축하면
# 대다수의 특성 가중치는 0으로 수렴(alpha 값에 따라서 조정)
print(lasso_model.coef_)

# - 학습
score = lr_model.score(X_train, y_train)
print(f'Train (LR) : {score}')
score = ridge_model.score(X_train, y_train)
print(f'Train (Ridge) : {score}')
score = lasso_model.score(X_train, y_train)
print(f'Train (Lasso) : {score}')

# - 테스트
score = lr_model.score(X_test, y_test)
print(f'Test (LR) : {score}')
score = ridge_model.score(X_test, y_test)
print(f'Test (Ridge) : {score}')
score = lasso_model.score(X_test, y_test)
print(f'Test (Lasso) : {score}')















