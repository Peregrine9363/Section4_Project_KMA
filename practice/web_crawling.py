from selenium import webdriver
from selenium.webdriver.ie.options import Options
import json
from flask import Flask
from flask import request
from datetime import date, datetime

app = Flask(__name__)

# get 공감수, 댓글수
@app.route('/get_sympathy',methods = ['POST','GET'])
def get_sympathy():

    sympathy = 0
    comment = 0
    return_obj = dict([])

    now =  datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return_obj["date"] = now

    try:
        # chromedriver
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        options.add_argument('window-size=1920x1080')
        options.add_argument("disable-gpu")
        driver = webdriver.Chrome('./chromedriver', options=options)

        try:
            blog_url = request.args.get('url')
            return_obj["blog_url"] = blog_url
        except:
            blog_url = 'unknown'

        # Load Page
        driver.get(url=blog_url)

        # 실제 페이지는 "https://blog.naver.com/PostView.naver?blogId=bizspringcokr&logNo=222638819175&redirect=Dlog&widgetTypeCall=true&directAccess=false"
        # iframe으로 호출된다.
        driver.switch_to.frame("mainFrame");

        # 공감
        #print("공감")
        sympathy_element = driver.find_element_by_css_selector('div.area_sympathy')
        em_elements = sympathy_element.find_elements_by_css_selector("em.u_cnt._count")
        #for i, em in enumerate(em_elements):
        #    print(i, "번째 em:", em.text)

        sympathy = int(em_elements[1].text)

        # 댓글
        #print("댓글")
        comment_elements = driver.find_element_by_css_selector('div.area_comment')
        em_elements = comment_elements.find_elements_by_css_selector("#floating_bottom_commentCount")
        #for i, em in enumerate(em_elements):
        #    print(i, "번째 em:", em.text)

        comment = int(em_elements[0].text)

    finally:
        driver.close()

    return_obj["sympathy"] = sympathy
    return_obj["comment"] = comment

    return return_obj

# 웹서비스로
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)