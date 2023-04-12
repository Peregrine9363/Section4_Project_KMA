import pandas as pd
import xgboost
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from category_encoders import OrdinalEncoder
from scipy.stats.distributions import uniform
from sklearn.metrics import roc_curve, roc_auc_score
import eli5
from eli5.sklearn import PermutationImportance
# from sklearn.metrics import plot_confusion_matrix 왜 오류?
import sklearn.metrics as metrics
import matplotlib.font_manager as fm

df = pd.read_excel("Data_EDA\df.xlsx")

# 인코딩
# 날짜 : 01-01 -> 101, 12-31 -> 1231
# 요일 : 월~일 -> (1~7)
# 날씨 : (맑음, 구름 조금, 흐림, 구름 많음, 비, 눈, 눈/비, 소나기, 비/눈)
# 날씨 : (0, 1, 2, 3, 4, 5, 5, 6, 5)
df_ec = df.copy()

import re

# 날짜 인코딩 함수
def date_ec(df):
    pattern = re.compile(r'^(\d{2})-(\d{2})$')
    
    for i in range(len(df)):
        date = df.iloc[i, 0] # 날짜
        match = pattern.match(date)
        
        if match:
            int_data = int(match.group(1) + match.group(2))
            df.iloc[i, 0] = int_data
    
    # 데이터 타입 변경
    df = df.astype({'날짜' : 'int'})
    
    return df
            
# 요일 인코딩 함수
def day_ec(df):
    # unique값 딕셔너리로 변환
    replace_dict = {'월':1, '화':2, '수':3, '목':4, '금':5, '토':6, '일':7}
    
    # replace() 함수를 사용하여 치환
    df.iloc[:, 1] = df.iloc[:, 1].replace(replace_dict)
    
    # 데이터 타입 변경
    df = df.astype({'요일' : 'int'})
    
    return df

# 날씨 인코딩 함수
def weather_ec(df):
    # unique값 딕셔너리로 변환
    replace_dict = {'맑음':0, '구름 조금':1, '흐림':2, '구름 많음':3, '비':4,
                    '눈':5, '눈/비':6, '소나기':2, '비/눈':6}
    
    # replace() 함수를 사용하여 치환
    df.iloc[:, 2] = df.iloc[:, 2].replace(replace_dict)
    
    # 데이터 타입 변경
    df = df.astype({'날씨' : 'int'})
    
    return df

df_ec = date_ec(df_ec)
df_ec = day_ec(df_ec)
df_ec = weather_ec(df_ec)

print(df_ec.dtypes)

# 훈련, 검증, 시험 세트 분리
target = '총계'
feature = df_ec.columns.drop(target)

train, test = train_test_split(df_ec, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.2, random_state=42)
print(train.shape, val.shape, test.shape)

def divide_data(df_ec):

    X = df_ec[feature]
    y = df_ec[target]
    
    return X, y

X_train, y_train = divide_data(train)
X_val, y_val = divide_data(val)
X_test, y_test = divide_data(test)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

# 기준 모델(회귀) : 타겟의 평균값
baseline = df_ec[target].mean()
baseline

# 모델 학습
from xgboost import XGBRegressor
from scipy.stats import uniform
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder

def fit(X_train, y_train):
    pipeline = make_pipeline(
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            )
        )
    
    params = {
    "xgbregressor__n_estimators" : [50, 100, 150, 200, 250, 300, 350],    
    "xgbregressor__learning_rate" : [0.01, 0.05, 0.1, 0.2, 0.3],
    "xgbregressor__max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "xgbregressor__min_child_weight": [1, 2, 4, 8, 16, 32, 64, 128],
    "xgbregressor__colsample_bytree": uniform(
        loc=0.5, scale=0.5
        ),  # 0.5 ~ 1 사이의 uniform 분포로 범위를 지정해 줍니다.
    }
    
    clf = RandomizedSearchCV(
        pipeline,
        param_distributions=params,
        scoring="neg_mean_squared_error",
        n_iter=10,
        cv=5,
        verbose=5,
        random_state=42,
        )
    
    clf.fit(X_train, y_train)
    print("Optimal Hyperparameter:", clf.best_params_)
    print("MSE:", -clf.best_score_)

    return clf

from sklearn.metrics import mean_squared_error, r2_score

clf = fit(X_train, y_train) # best_estimator

# 예측 및 성능 지표 계산 함수
def predict(X, y, best_estimator):
    y_pred = best_estimator.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

    return y_pred

y_val_pred = predict(X_val, y_val, clf)
y_test_pred = predict(X_test, y_test, clf)

# 기준값 성능 지표 계산
array_length = 365
baseline = np.full(array_length, baseline)

mse = mean_squared_error(y_test, baseline)
r2 = r2_score(y_test, baseline)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# 모델 저장
best_model = clf.best_estimator_.named_steps['xgbregressor']
best_model.save_model('model.bst')