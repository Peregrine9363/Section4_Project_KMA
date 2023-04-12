from selenium import webdriver

# 웹 드라이버 로드
driver = webdriver.Chrome('practice\chromedriver.exe')

# 해당 페이지 접속
url = "https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36&tabNo=4"
driver.get(url)

# 원하는 연도와 월 선택
year_select = driver.find_element_by_id("startYear")
year_select.send_keys("2022")  # 원하는 연도 입력

month_select = driver.find_element_by_id("startMonth")
month_select.send_keys("01")  # 원하는 월 입력

# 조회 버튼 클릭
search_button = driver.find_element_by_xpath("//button[contains(text(),'조 회')]")
search_button.click()