from selenium import webdriver
from bs4 import BeautifulSoup
import chromedriver_autoinstaller
import time
import random
import pandas as pd

class coin_crawl_his:
  
  # 종목명을 받음
  def __init__(self, title):
    self.title = title # 종목 명
    self.driver = None
    self.trs = None
    self.frame = None

  # chromedriver 자동 설치 밎 실행을 위한 코드
  def startDriver(self):    
    chromedriver_autoinstaller.install()
    self.driver = webdriver.Chrome()
    
    url = f'https://coinmarketcap.com/currencies/{self.title}/historical-data/'
    self.driver.get(url)

    self.driver.implicitly_wait(time_to_wait=10) # 로딩이 될때까지 대기  
    time.sleep(random.uniform(2,4)) # 로봇으로 인식하지 않도록 랜덤값으로 기다리기

  # 스크롤을 내려 페이지 내 데이터를 불러오는 함수
  def load_page_data(self):
    scroll_loaction = self.driver.execute_script("return document.body.scrollHeight")  # 현재 스크롤 위치

    while True:
      self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")

      bt = self.driver.find_element_by_xpath("//button[contains(text(),'Load More')]") # 불러오기 버튼의 요소를 가져온다
      self.driver.execute_script("arguments[0].click();",bt)

      self.driver.implicitly_wait(time_to_wait=10) # 로딩 대기
      time.sleep(2) # 버튼이 불러와지지 않는 것을 대비하여 sleep  

      scroll_height = self.driver.execute_script("return document.body.scrollHeight")  # 현재 스크롤 위치(내린 이후)

      if scroll_loaction == scroll_height:  # 내리기 전과 후가 같은 위치인 경우
        break
      else:
        scroll_loaction = self.driver.execute_script("return document.body.scrollHeight")

  # 정보가 있는 태그에서 값을 가져옴
  def get_data(self):
    html = self.driver.page_source # 모든 값을 다 불러온 이후의 소스 가져오기
    soup = BeautifulSoup(html, 'lxml')

    self.trs = soup.select('tr > td')
    
    # col명 - date, open, high, low, close, vol, marcket cap
    col = ['date', 'open', 'high', 'low', 'close', 'vol', 'marcket cap']

    # dataframe을 위한 형식 - dictionary 이용
    data = {
      'date' : [],
      'open' : [],
      'high' : [],
      'low' : [],
      'close' : [],
      'vol' : [],
      'marcket cap' : []
    }

    n = 0
    for i in self.trs:
      if n%7 == 0:
        data['date'].append(i.string)
      else:
        data[col[n%7]].append(i.string)
      n+=1
    self.frame = pd.DataFrame(data)

    return self.frame

  # driver 종료
  def quit_driver(self):
    self.driver.quit()
