import nltk
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence
import time
#라이브러리 선언부

stop_words = stopwords.words('english')
stemmer = PorterStemmer()
tweet_tokenizer = TweetTokenizer()

class sentimental_analysis:
    #데이터프레임을 입력받아 클래스초기화
    def __init__(self,twitter_data):
        self.twitter_data = twitter_data
        self.twitter_data = self.twitter_data[['date','tweet']]
        self.twitter_data['clean_text'] = np.NaN
    #vader,textblob,flair 옵션 입력
    def process(self,option): 
        start = time.time()
        self.twitter_data['tweet'] = self.twitter_data['tweet'].astype(str)
        self.twitter_data['clean_text'] = self.twitter_data['clean_text'].astype(str)
        #데이터 프레임 전처리 텍스트
        korean = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
        clean_tweet = []
        tweet = self.twitter_data
        for i in tweet.index:
            t = tweet._get_value(i,'tweet')
            tweet_list_corpus = re.sub(korean, '', str(t))
            clear_text = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",repl= ' ', string = tweet_list_corpus) # http로 시작되는 url
            clear_text = clear_text.replace('\n',' ').replace('\t',' ')
            clear_text = re.sub('RT @[\w_]+: ',' ', clear_text)
            # Hashtag 제거
            clear_text = re.sub('[#]+[0-9a-zA-Z_]+', ' ', clear_text)
            clear_text = re.sub('@','',clear_text)
            if option == 'textblob':
                clear_text = clear_text.lower()
            word_list = tweet_tokenizer.tokenize(clear_text)
            word_list = [word for word in word_list if word not in stop_words]
            word_list = [stemmer.stem(word) for word in word_list]
            text = ' '.join(word_list)
            if not text or text == np.NaN:
                self.twitter_data._set_value(i,'clean_text','000')
            else:
                self.twitter_data._set_value(i,'clean_text',text)
        if option == 'textblob':
            self.twitter_data['textblob'] = np.NaN
        else : 
            self.twitter_data['vader'] = np.NaN
            self.twitter_data['flair'] = np.NaN
        print("process time : ",time.time()-start)
    def sentimental_Textblob(self):
        start=time.time()
        result_tweet = self.twitter_data
        text_list = list(result_tweet['clean_text'])
        twitter_textblob_score = []
        for i in range(len(text_list)):
            blob = TextBlob(str(text_list[i]))
            for sentence in blob.sentences:
                if sentence == np.NaN:
                    sentece = '000'
                twitter_textblob_score.append(sentence.sentiment.polarity)
                self.twitter_data.loc[i,'textblob'] = twitter_textblob_score[i]
        print("textblob sentimental time : ",time.time()-start)
    def sentimental_vader(self):                    
        start = time.time()
        result_tweet = self.twitter_data
        #전처리된 csv파일 다시 reload
        text_list = result_tweet['clean_text']
        senti_analyzer = SentimentIntensityAnalyzer()
        twitter_vader_score = []
        for i in result_tweet.index:
            test = result_tweet._get_value(i,'clean_text')
            senti_scores = senti_analyzer.polarity_scores(test)
        #감정 분석 부분 
            self.twitter_data._set_value(i,'vader',senti_scores['compound'])
        #감정 분석 점수 데이터프레임에 추가
        print("vader sentimental time : ",time.time()-start)
    #플레어 감정분석기
    def sentimental_flair(self):
        start = time.time()
        senti_analyzer = TextClassifier.load('en-sentiment')
        result_tweet = self.twitter_data
        text_list = result_tweet['clean_text']
        n = (len(text_list))
        twitter_flair_score = []
        i = 0
        for i in range(0,n):
            sentence = Sentence(str(text_list[i]))
            senti_analyzer.predict(sentence)
            total_sen = sentence.labels[0]
            sign = 1 if total_sen.value == 'POSITIVE' else -1
            score = total_sen.score
            f_score = score*sign
            twitter_flair_score.append(f_score)
            self.twitter_data.loc[i,'flair'] = twitter_flair_score[i]
        print("flair sentimental time : ",time.time()-start)
    #감정분석 완료 이후 save함수를 통해서 데이터프레임 csv로 저장
    def save_csv(self):
        self.twitter_data.to_csv('senti_tweet.csv',mode = 'w')        
            #sentence 클래스 학습 이후 점수만 추출 예정
            
            
"""
bitcoin = pd.read_table("bitcoin_2022-04-14.txt",sep=",")
coin_tweet = sentimental_analysis(bitcoin)
coin_tweet.process('vader')
coin_tweet.sentimental_vader()
coin_tweet.sentimental_flair()
coin_tweet.save_csv()

모듈 사용 예시 소스코드
"""
