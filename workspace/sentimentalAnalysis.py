import nltk
import tqdm
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
import itertools
import tqdm

#라이브러리 선언부
def load_dict_contractions():
    return {
        "ain't":"is not",
        "amn't":"am not",
        "aren't":"are not",
        "can't":"cannot",
        "'cause":"because",
        "couldn't":"could not",
        "couldn't've":"could not have",
        "could've":"could have",
        "daren't":"dare not",
        "daresn't":"dare not",
        "dasn't":"dare not",
        "didn't":"did not",
        "doesn't":"does not",
        "don't":"do not",
        "e'er":"ever",
        "em":"them",
        "everyone's":"everyone is",
        "finna":"fixing to",
        "gimme":"give me",
        "gonna":"going to",
        "gon't":"go not",
        "gotta":"got to",
        "hadn't":"had not",
        "hasn't":"has not",
        "haven't":"have not",
        "he'd":"he would",
        "he'll":"he will",
        "he's":"he is",
        "he've":"he have",
        "how'd":"how would",
        "how'll":"how will",
        "how're":"how are",
        "how's":"how is",
        "I'd":"I would",
        "I'll":"I will",
        "I'm":"I am",
        "I'm'a":"I am about to",
        "I'm'o":"I am going to",
        "isn't":"is not",
        "it'd":"it would",
        "it'll":"it will",
        "it's":"it is",
        "I've":"I have",
        "kinda":"kind of",
        "let's":"let us",
        "mayn't":"may not",
        "may've":"may have",
        "mightn't":"might not",
        "might've":"might have",
        "mustn't":"must not",
        "mustn't've":"must not have",
        "must've":"must have",
        "needn't":"need not",
        "ne'er":"never",
        "o'":"of",
        "o'er":"over",
        "ol'":"old",
        "oughtn't":"ought not",
        "shalln't":"shall not",
        "shan't":"shall not",
        "she'd":"she would",
        "she'll":"she will",
        "she's":"she is",
        "shouldn't":"should not",
        "shouldn't've":"should not have",
        "should've":"should have",
        "somebody's":"somebody is",
        "someone's":"someone is",
        "something's":"something is",
        "that'd":"that would",
        "that'll":"that will",
        "that're":"that are",
        "that's":"that is",
        "there'd":"there would",
        "there'll":"there will",
        "there're":"there are",
        "there's":"there is",
        "these're":"these are",
        "they'd":"they would",
        "they'll":"they will",
        "they're":"they are",
        "they've":"they have",
        "this's":"this is",
        "those're":"those are",
        "'tis":"it is",
        "'twas":"it was",
        "wanna":"want to",
        "wasn't":"was not",
        "we'd":"we would",
        "we'd've":"we would have",
        "we'll":"we will",
        "we're":"we are",
        "weren't":"were not",
        "we've":"we have",
        "what'd":"what did",
        "what'll":"what will",
        "what're":"what are",
        "what's":"what is",
        "what've":"what have",
        "when's":"when is",
        "where'd":"where did",
        "where're":"where are",
        "where's":"where is",
        "where've":"where have",
        "which's":"which is",
        "who'd":"who would",
        "who'd've":"who would have",
        "who'll":"who will",
        "who're":"who are",
        "who's":"who is",
        "who've":"who have",
        "why'd":"why did",
        "why're":"why are",
        "why's":"why is",
        "won't":"will not",
        "wouldn't":"would not",
        "would've":"would have",
        "y'all":"you all",
        "you'd":"you would",
        "you'll":"you will",
        "you're":"you are",
        "you've":"you have",
        "Whatcha":"What are you",
        "luv":"love",
        "sux":"sucks"
        }

def load_dict_smileys(): 
    return {
        ":‑)":"smiley",
        ":-]":"smiley",
        ":-3":"smiley",
        ":->":"smiley",
        "8-)":"smiley",
        ":-}":"smiley",
        ":)":"smiley",
        ":]":"smiley",
        ":3":"smiley",
        ":>":"smiley",
        "8)":"smiley",
        ":}":"smiley",
        ":o)":"smiley",
        ":c)":"smiley",
        ":^)":"smiley",
        "=]":"smiley",
        "=)":"smiley",
        ":-))":"smiley",
        ":‑D":"smiley",
        "8‑D":"smiley",
        "x‑D":"smiley",
        "X‑D":"smiley",
        ":D":"smiley",
        "8D":"smiley",
        "xD":"smiley",
        "XD":"smiley",
        ":‑(":"sad",
        ":‑c":"sad",
        ":‑<":"sad",
        ":‑[":"sad",
        ":(":"sad",
        ":c":"sad",
        ":<":"sad",
        ":[":"sad",
        ":-||":"sad",
        ">:[":"sad",
        ":{":"sad",
        ":@":"sad",
        ">:(":"sad",
        ":'‑(":"sad",
        ":'(":"sad",
        ":‑P":"playful",
        "X‑P":"playful",
        "x‑p":"playful",
        ":‑p":"playful",
        ":‑Þ":"playful",
        ":‑þ":"playful",
        ":‑b":"playful",
        ":P":"playful",
        "XP":"playful",
        "xp":"playful",
        ":p":"playful",
        ":Þ":"playful",
        ":þ":"playful",
        ":b":"playful",
        "<3":"love"
        }
 
def strip_accents(text):
    if 'ø' in text or  'Ø' in text:
        return text   
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

    
stop_words = stopwords.words('english')
stemmer = PorterStemmer()
tweet_tokenizer = TweetTokenizer()

class sentimental_analysis:
    #데이터프레임을 입력받아 클래스초기화
    def __init__(self,twitter_data):
        self.twitter_data = twitter_data
        if not('clean_text' in twitter_data.columns) : 
            self.twitter_data = self.twitter_data[['date','tweet']]
            self.twitter_data['clean_text'] = np.NaN
        else : 
            self.twitter_data = self.twitter_data[['date','tweet','clean_text']]

    #vader,textblob,flair 옵션 입력
    def process(self,option): 
        start = time.time()
        self.twitter_data['tweet'] = self.twitter_data['tweet'].astype(str)
        self.twitter_data['clean_text'] = self.twitter_data['clean_text'].astype(str)
        #데이터 프레임 전처리 텍스트
        korean = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
        clean_tweet = []
        tweet = self.twitter_data
        for i in tqdm.notebook.tqdm(tweet.index):
            t = tweet._get_value(i,'tweet')
            tweet_list_corpus = re.sub(korean, '', str(t))
            clear_text = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                                repl= ' ', string = tweet_list_corpus) 
            # http로 시작되는 url
            clear_text = clear_text.replace('\n',' ').replace('\t',' ')
            clear_text = re.sub('RT @[\w_]+: ',' ', clear_text)
            #한글, 주소, 엔터키, 리트윗 삭제
            clear_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", clear_text).split())
            clear_text = ' '.join(re.sub("(\w+:\/\/\S+)", " ", clear_text).split())
            clear_text = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", clear_text).split())
            #해쉬태그 및 태그 , 주소, 구두점 제거
            CONTRACTIONS = load_dict_contractions()
            clear_text = clear_text.replace("’","'")
            words = clear_text.split()
            reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
            clear_text = " ".join(reformed)
            clear_text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(clear_text))
            #영어 표현 정규화, 단어 그룹핑을통한 정규화
            
            SMILEY = load_dict_smileys()  
            words = clear_text.split()
            reformed = [SMILEY[word] if word in SMILEY else word for word in words]
            clear_text = " ".join(reformed)
            #이모티콘 텍스트화
            clear_text = strip_accents(clear_text)
            clear_text = clear_text.replace(":"," ")
            clear_text = ' '.join(clear_text.split())
            if option == 'textblob':
                clear_text = clear_text.lower()
            
            word_list = tweet_tokenizer.tokenize(clear_text)
            text = ' '.join(word_list)
            if text == "":
                self.twitter_data._set_value(i,'clean_text',np.NaN)
            else:
                self.twitter_data._set_value(i,'clean_text',text)
        self.twitter_data.dropna(axis=0)
        if option == 'textblob':
            self.twitter_data['textblob'] = np.NaN
        elif option == 'vader' or option == 'flair':   
            self.twitter_data['vader'] = np.NaN
            self.twitter_data['flair'] = np.NaN
        print("process time : ",time.time()-start)
    def sentimental_with_textblob(self):
        start=time.time()
        for i in tqdm.notebook.tqdm(self.twitter_data.index):
            tweet_data = self.twitter_data._get_value(i,'clean_text')
            blob = TextBlob(str(tweet_data))
            #감정 분석 부분
            for sentence in blob.sentences:
                self.twitter_data._set_value(i,'textblob',sentence.sentiment.polarity) 
        print("textblob sentimental time : ",time.time()-start)
    def sentimental_with_vader(self):                    
        start = time.time()
        #전처리된 csv파일 다시 reload
        senti_analyzer = SentimentIntensityAnalyzer()
        for i in tqdm.notebook.tqdm(self.twitter_data.index):
            tweet_data = self.twitter_data._get_value(i,'clean_text')
            senti_scores = senti_analyzer.polarity_scores(str(tweet_data))
        #감정 분석 부분 
            self.twitter_data._set_value(i,'vader',senti_scores['compound'])
        #감정 분석 점수 데이터프레임에 추가
        print("vader sentimental time : ",time.time()-start)
    #플레어 감정분석기
    def sentimental_with_flair(self):
        start = time.time()
        senti_analyzer = TextClassifier.load('en-sentiment')       
        text_list = self.twitter_data['clean_text']

        for i in tqdm.notebook.tqdm(self.twitter_data.index):
            text = str(self.twitter_data._get_value(i,'clean_text'))
            sentence = Sentence(text)
            senti_analyzer.predict(sentence)
            total_sen = sentence.labels[0]
            sign = 1 if total_sen.value == 'POSITIVE' else -1
            score = total_sen.score
            f_score = score*sign
            self.twitter_data._set_value(i,'flair',f_score)
        print("flair sentimental time : ",time.time()-start)
        #감정분석 완료 이후 save함수를 통해서 데이터프레임 csv로 저장
    
    def save_csv(self,file_name):
        self.twitter_data.to_csv(file_name+'.csv',mode = 'w')        
            
"""
bitcoin = pd.read_table("bitcoin_2022-04-14.txt",sep=",")
coin_tweet = sentimental_analysis(bitcoin)
coin_tweet.process('vader')
coin_tweet.sentimental_with_vader()
coin_tweet.sentimental__with_flair()
coin_tweet.save_csv()

모듈 사용 예시 소스코드
"""