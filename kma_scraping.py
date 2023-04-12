from selenium import webdriver
import pandas as pd
import time

# 웹 드라이버 로드
driver = webdriver.Chrome("chromedriver_win32\chromedriver.exe")

# 해당 페이지 접속
url = "https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36&tabNo=4"
driver.get(url)

# 원하는 연도와 월 선택 후 조회 버튼 클릭하는 함수
def select(year, month):
    year_select = driver.find_element_by_id("startYear")
    year_select.send_keys(year)  # 원하는 연도 입력
    
    # if는 버그 해결을 위해 추가(원인은 아직 모르겠음)
    if month != '11':
        month_select = driver.find_element_by_id("startMonth")
        month_select.send_keys(month)  # 원하는 월 입력
    else:
        month_select = driver.find_element_by_id("startMonth")
        month_select.send_keys(month)
        month_select.send_keys(month)
    
    search_button = driver.find_element_by_css_selector("button.btn")
    search_button.click() # 조회 버튼 클릭

# 데이터프레임 생성
index_name = range(0, 31)
column_name = range(0, 12)
df = pd.DataFrame(index = index_name)
df = pd.DataFrame(df, columns = column_name)

# copy로 해야 데이터 변경이 연동되지 않음
df_2018 = df.copy()
df_2019 = df.copy()
df_2020 = df.copy()
df_2021 = df.copy()
df_2022 = df.copy()
        
# month 데이터가 str타입
month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

# kma에서 날씨 데이터를 스크래핑하고 데이터프레임에 저장하는 함수
def kma_scraping(year, df):
    # 월 초기화
    m = 0
    for j in month:
        select(f'{year}', j)
        
        # 기온에 해당하는 요소를 CSS Selector로 검색하고 리스트로 반환
        cal_schedule = driver.find_elements_by_css_selector('div.cal-schedule')
        
        # 날짜 별 기온을 tp_list에 반환
        tp_list = []
        for schedule in cal_schedule:
            if schedule.text != '': # 데이터가 있는 날만 추가
                tp_list.append(schedule.text)
                print(schedule.text)
                
        # tp가 31일 단위가 되도록 padding하고 tp_list값 대입
        tp = [None] * 31
        
        for k in range(len(tp_list)):
            tp[k] = tp_list[k]
        
        # 데이터 프레임에 저장
        for d in range(0, 31):
            df.loc[d, m] = tp[d]
        m += 1

kma_scraping(2018, df_2018)
kma_scraping(2019, df_2019)
kma_scraping(2020, df_2020)
kma_scraping(2021, df_2021)
kma_scraping(2022, df_2022)

# 스크래핑한 데이터 csv로 저장
df_2018.to_csv("Data_EDA/2018_csv.csv")
df_2018.to_excel("Data_EDA/2018_w.xlsx", index=False)

df_2019.to_csv("Data_EDA/2019_csv.csv")
df_2019.to_excel("Data_EDA/2019_w.xlsx", index=False)

df_2020.to_csv("Data_EDA/2020_csv.csv")
df_2020.to_excel("Data_EDA/2020_w.xlsx", index=False)

df_2021.to_csv("Data_EDA/2021_csv.csv")
df_2021.to_excel("Data_EDA/2021_w.xlsx", index=False)

df_2022.to_csv("Data_EDA/2022_csv.csv")
df_2022.to_excel("Data_EDA/2022_w.xlsx", index=False)

# 엑셀 데이터 불러와서 취합하는 함수
def excel_concat(import_path, export_path):
    # 서울대공원 일별 관람객 엑셀 데이터 추출
    df_list = []
    for i in range(0, 12):
        df = pd.read_excel(import_path, sheet_name=i)
        df_list.append(df)
        
    # 각 연도별 기준으로 월별 데이터 통합
    concat_list = [
                    df_list[0], df_list[1], df_list[2], df_list[3],
                    df_list[4], df_list[5], df_list[6], df_list[7],
                    df_list[8], df_list[9], df_list[10], df_list[11]
                    ]
        
    data = pd.concat(concat_list, ignore_index=True)

    # 통합된 데이터 엑셀로 저장
    data.to_excel(export_path, index=False)

excel_concat("Data_EDA/2018_EDA.xls", "Data_EDA/2018_EDA_concat.xlsx")
excel_concat("Data_EDA/2019_EDA.xls", "Data_EDA/2019_EDA_concat.xlsx")
excel_concat("Data_EDA/2020_EDA.xlsx", "Data_EDA/2020_EDA_concat.xlsx")
excel_concat("Data_EDA/2021_EDA.xlsx", "Data_EDA/2021_EDA_concat.xlsx")
excel_concat("Data_EDA/2022_EDA.xlsx", "Data_EDA/2022_EDA_concat.xlsx")
    
"""
# 서울대공원 일별 관람객 엑셀 데이터 추출
df_list = []
for i in range(0, 12):
    df = pd.read_excel('Data_EDA/2018_EDA.xls', sheet_name=i)
    df_list.append(df)

# 각 연도별 기준으로 월별 데이터 통합
concat_list = [
                df_list[0], df_list[1], df_list[2], df_list[3],
                df_list[4], df_list[5], df_list[6], df_list[7],
                df_list[8], df_list[9], df_list[10], df_list[11]
                ]
data = pd.concat(concat_list, ignore_index=True)

# 통합된 데이터 엑셀로 저장
data.to_excel('Data_EDA/2018_EDA_concat.xlsx', index=False)
"""