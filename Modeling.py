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
"""
# 폰트 경로 지정
font_name = 'Malgun Gothic'

# 폰트 프로퍼티 설정
font_prop = fm.FontProperties(fname=font_path, size=12)

# Matplotlib 전역 폰트 설정
plt.rc('font', family=font_name)

# 데이터 분포 확인
sns.histplot(data=df, x="총계", y="요일").set(title='총 관람객 수')
plt.show()

# 강수량과 총계 상관관계 확인
df_check = df[['강수량(mm)', '총계']]
df_corr = np.corrcoef(df_check['강수량(mm)'], df_check['총계'])[0, 1]
print(round(df_corr, 2))

# 타겟 분포 시각화
sns.set_style("whitegrid")
sns.histplot(df["총계"]).set(title='관람객 분포')
"""

# 인코딩
from sklearn.preprocessing import OrdinalEncoder

def ORE(data):
    # 데이터를 2D 배열로 변환
    data_2d = data.values.reshape(-1, 1)

    # Ordinal 인코더 객체 생성 및 학습
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(data_2d)
    
    # 변환된 정수 레이블 출력
    encoded_data = ordinal_encoder.transform(data_2d)
    
    return encoded_data

df_encoded = df.copy()
df_encoded['날짜'] = ORE(df['날짜'])
df_encoded['요일'] = ORE(df['요일'])
df_encoded['날씨'] = ORE(df['날씨'])

# 훈련, 검증, 시험 세트 분리
target = '총계'
feature = df_encoded.columns.drop(target)

train, test = train_test_split(df_encoded, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.2, random_state=42)
print(train.shape, val.shape, test.shape)

def divide_data(df_encoded):

    X = df_encoded[feature]
    y = df_encoded[target]
    
    return X, y

X_train, y_train = divide_data(train)
X_val, y_val = divide_data(val)
X_test, y_test = divide_data(test)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

# 기준 모델(회귀) : 타겟의 평균값
baseline = df_encoded[target].mean()
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

"""
def predict(X, y, best_estimator):
    y_pred = best_estimator.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

    return y_pred
"""

clf = fit(X_train, y_train)

# 예측 및 성능 지표 계산
y_val_pred = clf.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print("Mean Squared Error on validation set:", mse)
print("R^2 Score on validation set:", r2)

y_test_pred = clf.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
print("Mean Squared Error on test set:", mse)
print("R^2 Score on test set:", r2)

# 기준값 성능 지표 계산
array_length = 365
baseline = np.full(array_length, baseline)

mse = mean_squared_error(y_test, baseline)
r2 = r2_score(y_test, baseline)
print("Mean Squared Error on baseline set:", mse)
print("R^2 Score on baseline set:", r2)