import requests

from requests.exceptions import HTTPError

url = 'https://google.com'

try:
    resp = requests.get(url)

    resp.raise_for_status()
except HTTPError as Err:
    print('HTTP 에러가 발생했습니다.')
except Exception as Err:
    print('다른 에러가 발생했습니다.')
else:
    print('성공')
    
from bs4 import BeautifulSoup

url = 'https://google.com'
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')

dog_element = soup.find(id='dog')

cat_elements = soup.find_all(class_='cat')

for cat_el in cat_elements:
    cat_el.find(class_='fish')
    
cat_div_elements = soup.find_all('div', class_='cat')

soup.find_all(string='raining')

soup.find_all(string=lambda text: 'raining' in text.lower())

soup.find_all('h3', string='raining')

cat_el = soup.find('p', class_='cat')