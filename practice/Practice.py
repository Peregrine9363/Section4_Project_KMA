import re
import requests
from bs4 import BeautifulSoup
import pprint

url = 'https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?'
params = {
    'pgmNo': '36',
    'tabNo': '4',
    'year': '2022',
    'dateGb': 'daily',
    'stnIds': '108',  # 관측소 ID
    'searchTerm': '년도별조회',
    'startDt': '20220101',  # 시작일
    'endDt': '20221231'  # 종료일
}
res = requests.get(url, params=params)
html = res.text

soup = BeautifulSoup(html, 'html.parser')

# 날씨 데이터가 포함된 테이블 태그 찾기 (테이블 태그?)
table = soup.find_all('table', {'class': 'tbl'})

table_date = table[1]

pprint.pprint(table_date)

# 테이블의 th 태그에서 컬럼명 추출하기 (th가 뭐지?)
columns = []
for th in table.find_all('th'):
    columns.append(th.text.strip())
    
pprint.pprint(columns)

# 테이블의 td 태그에서 데이터 추출하기 (td는?)
data = []
for tr in table.find_all('tr')[1:]:
    row = []
    for td in tr.find_all('td'):
        row.append(td.text.strip())
    data.append(row)
    
pprint.pprint(data)

# selenium으로 크롬 브라우저창 열기
from selenium import webdriver

driver = webdriver.Chrome('practice\chromedriver.exe')
url = 'https://www.google.com'
driver.get(url)

# Load Page
# chrome을 띄워 네이버 블로그 페이지를 연다.
driver.get(url='https://blog.naver.com/bizspringcokr/222638819175')

# 현재 URL을 출력
print(driver.current_url)

driver.close()

# GUI가 아닌 환경에서도 작동할 수 있도록 하기
from selenium import webdriver
from selenium.webdriver.ie.options import Options

# chromedriver
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")
driver = webdriver.Chrome('practice\chromedriver.exe', options=options)

# Load Page
driver.get(url='https://blog.naver.com/bizspringcokr/222638819175')

print(driver.current_url)

driver.close()