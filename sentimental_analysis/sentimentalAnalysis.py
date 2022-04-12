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
        
#라이브러리 선언부

stop_words = stopwords.words('english')
stemmer = PorterStemmer()
tweet_tokenizer = TweetTokenizer()

class sentimental_analysis:
    def __init__(self,twitter_data):
        self.twitter_data = twitter_data
    def process(self):        
        self.twitter_data['clean_text'] = np.NaN
        self.twitter_data['vader'] = np.NaN
        self.twitter_data['textblob'] = np.NaN
        #데이터 프레임 전처리 텍스트 + 점수 목록 추가 
        kor_text = 'asdfasdfㅋㅌㅊㅍㅋㅌㅊㅍ1234234가나다라*@*@#*#@*'
        clean_tweet = []
        x = 0
        tweet = list(self.twitter_data.loc[2000000:2500000]['text'])
        for tweet_data in tweet:
            tweet_list_corpus = tweet_data.lower()
            # URL 제거
            clear_text = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",repl= ' ', string = tweet_list_corpus) # http로 시작되는 url
            clear_text = clear_text.replace('\n',' ').replace('\t',' ')
            clear_text = re.sub('RT @[\w_]+: ',' ', clear_text)
            # Hashtag 제거
            clear_text = re.sub('[#]+[0-9a-zA-Z_]+', ' ', clear_text)

            # 쓰레기 단어 제거
            clear_text = re.sub('[&]+[a-z]+', ' ', clear_text)
            clean_text = re.sub(kor_text, ' ', clear_text)
            # 특수문자 제거
            #clear_text = re.sub('[^0-9a-zA-Z]', ' ', clear_text)
            word_list = tweet_tokenizer.tokenize(clear_text)
            word_list = [word for word in word_list if word not in stop_words]
            word_list = [stemmer.stem(word) for word in word_list]
            text = ' '.join(word_list)
            self.twitter_data.at[x,'clean_text'] = text
            x = x + 1
        #전처리 부분 -> 함수화  + 사용하는 감정 분석기에 맞게 수정 예정

        twitter_csv = self.twitter_data[2000000:2500000]
        twitter_csv.to_csv("clean_tweets.csv",mode = 'a',header=False)
        #총 데이터가 약 250만개로 50만개씩 전처리 + 저장 방식으로 수행
        return twitter_csv
    
    def sentimental_textblob(result_tweet):
        result_tweet = pd.read_csv("clean_tweets.csv")
        #전처리된 csv파일 다시 reload
        text_list = list(result_tweet['clean_text'])
        twitter_textblob_score = []
        for text in text_list:
            blob_text = str(text)
            blob = TextBlob(blob_text)
            for sentence in blob.sentences:
                #print(sentence.sentiment.polarity, sentence.sentiment.subjectivity, sentence)
                twitter_textblob_score.append(sentence.sentiment.polarity)
            for i in range(len(result_tweet)):
                result_tweet.at[i,'textblob'] = twitter_textblob_score[i]
        #감정 분석 점수 데이터프레임에 추
        result_tweet.to_csv("result_tweeets.csv",mode = 'w')
    
    def sentimental_vader(result_tweet):            
        result_tweet = pd.read_csv("clean_tweets.csv")
        #전처리된 csv파일 다시 reload
        text_list = list(result_tweet['clean_text'])
        senti_analyzer = SentimentIntensityAnalyzer()
        twitter_vader_score = []
        for i in range(0,len(text_list)):
            test = str(text_list[i])
            senti_scores = senti_analyzer.polarity_scores(test)
            twitter_vader_score.append(senti_scores['compound'])
        #감정 분석 부분 
        for i in range(len(result_tweet)):
            result_tweet.at[i,'vader'] = twitter_vader_score[i]
        #감정 분석 점수 데이터프레임에 추가
        result_tweet.to_csv("result_tweeets.csv",mode = 'w')
        