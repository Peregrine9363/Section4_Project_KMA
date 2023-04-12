import pandas as pd
import re

# 입장객 데이터 추출
df_18 = pd.read_excel("Data_EDA/2018_EDA_concat.xlsx")
df_19 = pd.read_excel("Data_EDA/2019_EDA_concat.xlsx")
df_20 = pd.read_excel("Data_EDA/2020_EDA_concat.xlsx")
df_21 = pd.read_excel("Data_EDA/2021_EDA_concat.xlsx")
df_22 = pd.read_excel("Data_EDA/2022_EDA_concat.xlsx")

# 날씨 데이터 추출
w_18 = pd.read_excel("Data_EDA/2018_w.xlsx")
w_19 = pd.read_excel("Data_EDA/2019_w.xlsx")
w_20 = pd.read_excel("Data_EDA/2020_w.xlsx")
w_21 = pd.read_excel("Data_EDA/2021_w.xlsx")
w_22 = pd.read_excel("Data_EDA/2022_w.xlsx")

# 평균기온(℃) 추출 함수
def avg_temp(w):
    at = w.copy()
    for i in range(len(w)):
        for j in range(12):
            data = w.iloc[i, j]
            # 데이터가 NaN인지 확인 (NaN이면 split안됌)
            if not pd.isna(data): # data가 NaN이 아닌 경우에만 처리
                lines = data.split("\n") # \n 문자열 처리
                avg_temp_pattern = r"평균기온\(℃\):([+-]?\d+(\.\d+)?)"
                avg_temp = None
                for line in lines: # lines의 요소마다 검색
                    match = re.search(avg_temp_pattern, line)
                    if match:
                        avg_temp = float(match.group(1)) # ([+-]?\d+(\.\d+)?) 추출
                        break
            else:
                avg_temp = None
        
            at.iloc[i, j] = avg_temp
    return at

at_18 = avg_temp(w_18)
at_19 = avg_temp(w_19)
at_20 = avg_temp(w_20)
at_21 = avg_temp(w_21)
at_22 = avg_temp(w_22)

# 일강수량(mm) 추출 함수
def rainfall(w):
    rf = w.copy()
    for i in range(len(w)):
        for j in range(12):
            data = w.iloc[i, j]
            # 데이터가 NaN인지 확인 (NaN이면 split안됌)
            if not pd.isna(data): # data가 NaN이 아닌 경우에만 처리
                lines = data.split("\n") # \n 문자열 처리
                rainfall_pattern = r"일강수량\(mm\):([+-]?\d+(\.\d+)?)"
                rainfall = None
                for line in lines: # lines의 요소마다 검색
                    match = re.search(rainfall_pattern, line)
                    if match:
                        rainfall = float(match.group(1)) # ([+-]?\d+(\.\d+)?) 추출
                        break
            else:
                rainfall = None
        
            rf.iloc[i, j] = rainfall
    return rf

rf_18 = rainfall(w_18)
rf_19 = rainfall(w_19)
rf_20 = rainfall(w_20)
rf_21 = rainfall(w_21)
rf_22 = rainfall(w_22)

# row가 연도 전체 일로 구성되도록 변환 함수(평균기온)
def day_concat_at(df):
    df_365 = pd.DataFrame(index = range(365))
    df_365 = pd.DataFrame(df_365, columns = ['avg_temp'])
    k = 0
    for j in range(12):
        for i in range(len(df)):
            day = df.iloc[i, j] # 각 일 별 데이터
            if not pd.isna(day): # 결측값이 아니면 1열에 추가
                df_365.loc[k] = [day]
                k += 1
            else :
                k += 1
    return df_365

at_18 = day_concat_at(at_18)
at_19 = day_concat_at(at_19)
at_20 = day_concat_at(at_20)
at_21 = day_concat_at(at_21)
at_22 = day_concat_at(at_22)

# row가 연도 전체 일로 구성되도록 변환 함수(강수량)
def day_concat_rf(df):
    df_365 = pd.DataFrame(index = range(365))
    df_365 = pd.DataFrame(df_365, columns = ['rainfall'])
    k = 0
    for j in range(12):
        for i in range(len(df)):
            day = df.iloc[i, j] # 각 일 별 데이터
            if not pd.isna(day): # 결측값이 아니면 1열에 추가
                df_365.loc[k] = [day]
                k += 1
            else :
                k += 1
    return df_365

rf_18 = day_concat_rf(rf_18)
rf_19 = day_concat_rf(rf_19)
rf_20 = day_concat_rf(rf_20)
rf_21 = day_concat_rf(rf_21)
rf_22 = day_concat_rf(rf_22)

# 일별 관람객 데이터에 날씨 데이터 추가
df_18['평균기온(℃)'] = at_18
df_19['평균기온(℃)'] = at_19
df_20['평균기온(℃)'] = at_20
df_21['평균기온(℃)'] = at_21
df_22['평균기온(℃)'] = at_22

df_18['강수량(mm)'] = rf_18
df_19['강수량(mm)'] = rf_19
df_20['강수량(mm)'] = rf_20
df_21['강수량(mm)'] = rf_21
df_22['강수량(mm)'] = rf_22

# 데이터셋을 합침
concat_list = [df_18, df_19, df_20, df_21, df_22]
df = pd.concat(concat_list, ignore_index=True)

# df 원본 데이터 저장
df.to_csv("Data_EDA/df_backup_csv.csv")
df.to_excel("Data_EDA/df_backup.xlsx", index=False)

# EDA
# 날짜 월,일만 구분
from datetime import datetime

for i in range(len(df)):
    date_obj = df.iloc[i, 0]
    date_str = date_obj.strftime("%Y-%m-%d") # 시계열 데이터를 str로 바꿈
    new_date_str = re.sub(r'\d{4}-', '', date_str)
    df.iloc[i, 0] = new_date_str
    
# 날씨 결측치 채움(강수량 데이터를 활용)
for i in range(len(df)):
    if pd.isna(df.iloc[i, 2]) and pd.isna(df.iloc[i, -1]): # 강수량이 없으면 0으로 하고 날씨는 맑음
        df.iloc[i, 2] = '맑음'
        df.iloc[i, -1] = 0
        
        # 강수량에 따른 날씨
    elif pd.isna(df.iloc[i, 2]) and df.iloc[i, -1] < 2.5:
        df.iloc[i, 2] = '구름 조금'
    elif pd.isna(df.iloc[i, 2]) and df.iloc[i, -1] >= 2.5 and df.iloc[i, -1] < 6.5:
        df.iloc[i, 2] = '구름 많음'
    elif pd.isna(df.iloc[i, 2]) and df.iloc[i, -1] >= 6.5 and df.iloc[i, -1] < 20:
        df.iloc[i, 2] = '흐림'
    elif pd.isna(df.iloc[i, 2]) and df.iloc[i, -1] >= 20:
        df.iloc[i, 2] = '비'
        
        # 날씨에 따른 강수량
    elif df.iloc[i, 2] == '비' and pd.isna(df.iloc[i, -1]):
        df.iloc[i, -1] = 20
    elif df.iloc[i, 2] == '흐림' and pd.isna(df.iloc[i, -1]):
        df.iloc[i, -1] = 6.5
    elif df.iloc[i, 2] == '구름 조금' and pd.isna(df.iloc[i, -1]):
        df.iloc[i, -1] = 1.25
    elif df.iloc[i, 2] == '맑음' and pd.isna(df.iloc[i, -1]):
        df.iloc[i, -1] = 0
    else:
        df.iloc[i, -1] = 0

# 나머지 결측치 대체
df.iloc[:,-2].fillna(df.iloc[:, -2].mean(), inplace=True)

# 불필요한 column 삭제
df.drop('유료합계', axis=1, inplace=True)
df.drop('무료합계', axis=1, inplace=True)

# 최종 데이터셋 저장
df.to_csv("Data_EDA/df.csv", index=False)
df.to_excel("Data_EDA/df.xlsx", index=False)