from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import random

title = input("input title : ") # 종목 명 입력

driver = webdriver.Chrome('C:/Users/epicl/Documents/chdr/chromedriver.exe')

url = f'https://coinmarketcap.com/currencies/{title}/historical-data/'
driver.get(url)
# /table/tbody/tr/td
# element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "sc-16r8icm-0 jKrmxw container")))
# try:
#   element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "/table/tbody/tr/td")))
# finally:
#   driver.quit()
driver.implicitly_wait(time_to_wait=10) # 로딩이 될때까지 대기  
time.sleep(random.uniform(2,4)) # 로봇으로 인식하지 않도록 랜덤값으로 기다리기
# html = driver.page_source
# soup = BeautifulSoup(html, 'lxml')


scroll_loaction = driver.execute_script("return document.body.scrollHeight")  # 현재 스크롤 위치

while True:
  driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
  # /body/div/div/div/div[1]/div/div[2]/div/div/p/button
  # driver.find_element_by_class_name('x0o17e-0 DChGS').click()
  bt = driver.find_element_by_xpath("//button[contains(text(),'Load More')]") # 불러오기 버튼의 요소를 가져온다
  bt.click()  # 불러오기 버튼 클릭

  driver.implicitly_wait(time_to_wait=10) # 로딩 대기
  time.sleep(2) # 버튼이 불러와지지 않는 것을 대비하여 sleep  

  scroll_height = driver.execute_script("return document.body.scrollHeight")  # 현재 스크롤 위치(내린 이후)

  if scroll_loaction == scroll_height:  # 내리기 전과 후가 같은 위치인 경우
    break
  else:
    scroll_loaction = driver.execute_script("return document.body.scrollHeight")

html = driver.page_source # 모든 값을 다 불러온 이후의 소스 가져오기
soup = BeautifulSoup(html, 'lxml')

trs = soup.select('tr > td') 


# time.sleep(3)


info = list()
cat = ['date', 'open', 'high', 'low', 'close', 'vol', 'marcket cap']
# 7개 - date, open, high, low, close, vol, marcket cap

tn = 0
for i in trs:
  if tn%7 == 0:
    if tn == 0: # 첫 값
      temp = dict()
    elif temp['date'] != i.string:
      info.append(temp)
      temp = dict()
    temp[cat[0]] = i.string
  else:
    temp[cat[tn%7]] = i.string
  tn += 1

for i in info:
  print(i)

driver.quit()