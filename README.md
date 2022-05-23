# NaturalProcessing
Natural Process Algorithms for gree/fear analysis and database
- Developer : Kim, Doyoung, Kim Juwon

# Explanation
## 1. Python을 이용한 코인&트위터 크롤러 API 
파이썬을 이용하여 구현한 API입니다. 코인 크롤러 API는 종목을 입력하면 해당 종목에 대한 전체 시가, 종가, 상한가, 하한가, 시가 총액 등 해당 종목의 가격 히스토리를 날짜별로 수집합니다. 트위터 크롤러는 특정 가상화폐 종목을 토대로하여 트윗을 검색한 뒤 해당 기간 트윗의 텍스트, 작성ID, 작성시간, likes 숫자 등 트윗 관련 데이터를 크롤링을 할 수 있도록 API를 구현해주었습니다. 

![image](https://user-images.githubusercontent.com/81630351/169848576-5a5243cc-0012-4270-bf57-44cbd8a1a9e8.png)
![image](https://user-images.githubusercontent.com/81630351/169848602-7aa76de5-ae73-4c1b-8ef0-23dda7eda8cd.png)


<Twitter 크롤러 API(좌측)과 크롤링한 데이터(우측)>

![image](https://user-images.githubusercontent.com/81630351/169848803-f5f24a74-a690-4829-8aba-0bd1ecda94db.png)
![image](https://user-images.githubusercontent.com/81630351/169848827-cd142f39-d1c5-4465-b3dc-397e0ed2e116.png)


<Coin Price Crawler API(좌측)과 크롤링한 데이터(우측)>

## 2. 자연어처리 및 감성분석 모듈(자연어처리-감성분석 파이프라인)
파이썬을 이용하여 감성분석 모듈입니다. 영어를 전처리 및 감성분석을 실행해주는 모듈입니다. 전처리 함수를 통해 트윗 텍스트에 있는 불용어, 해시태그, URL등을 제거하고, 이모티콘을 텍스트화하여 감성분석을 할 수 있도록 텍스트를 전처리 합니다. 이후 옵션을 선택하여 규칙 기반인 textblob, vader라이브러리를 이용하여 감성 분석을 진행하거나, 딥러닝 기반의 flair라이브러리를 이용하여 감성분석을 진행할 수 있도록 구현해주었습니다.

![image](https://user-images.githubusercontent.com/81630351/169850827-a6c2388d-3675-4f46-ba68-06ebfcf8b9ed.png)
![image](https://user-images.githubusercontent.com/81630351/169850843-85955b1a-177e-4af3-a5cb-55059907798f.png)


<감성분석모듈(좌측)과 Textblob, Vader감성분석을 마친 데이터(우측)>

## 3 코인 가격 예측 모델(Greed_Fear_Model)
다중 선형 회귀(Multi-Linear Regression)모델입니다. 감성분석이 된 트윗의 자연어 데이터와 해당 날짜의 '종가'를 전달받아 다음 날의 '종가'를 예측하는 모델입니다. 모델을 통해 다음 날짜의 예측 가격 뿐만이 아니라, 가격의 등락을 Greed_Fear_Index로 보여줍니다. Python의 statsmodel.api의 OLS를 이용하여 다중 선형 회귀 모델을 구현해주었습니다. 

![image](https://user-images.githubusercontent.com/81630351/169851076-ba6d4667-174f-43b8-85ca-09bb2f3575d0.png)
![image](https://user-images.githubusercontent.com/81630351/169851092-faeed42d-1046-4d20-8ce2-c2df254bdbef.png)


<모델 내용(좌측) 및 종목별로 크롤링한 데이터의 예측 정확도(우측)>
