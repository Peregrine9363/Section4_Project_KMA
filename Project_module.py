from selenium import webdriver

# 원하는 연도와 월 선택 후 조회 버튼 클릭하는 함수
def select(year, month):
    # 웹 드라이버 로드
    driver = webdriver.Chrome("chromedriver_win32\chromedriver.exe")
    
    # 해당 페이지 접속
    url = "https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36&tabNo=4"
    driver.get(url)

    year_select = driver.find_element_by_id("startYear")
    year_select.send_keys(year)  # 원하는 연도 입력
    
    month_select = driver.find_element_by_id("startMonth")
    month_select.send_keys(month)  # 원하는 월 입력
    
    search_button = driver.find_element_by_css_selector("button.btn")
    search_button.click() # 조회 버튼 클릭

def test():
    return print("good")

# 날짜 딕셔너리 생성
keys = []
for day in range(1, 32):
    keys.append(day)

days = {key: value for key, value in dict.fromkeys(keys).items()}