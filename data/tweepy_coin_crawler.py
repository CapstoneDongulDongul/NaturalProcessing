import tweepy
import csv

class tweepy_coin_crawler:

    # 키를 받아주는 함수
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.api = None

    # tweepy를 연결시켜주는 함수
    def tweepy_connect(self):
        auth = tweepy.OAuth1UserHandler(self.consumer_key, self.consumer_secret, 
                                        self.access_token, self.access_token_secret)
        self.api = tweepy.API(auth)
        return self

    # 현재 날짜로부터 일주일 전 날짜만 입력 가능합니다.
    def tweepy_crawler_with_7_days(self, location, keyword):
        file = open("./twitter.csv",'w', newline='')        # csv로 저장
        wr = csv.writer(file) # twitter 검색 cursor 선언
        year = input("연도를 입력하세요(2022년이후):")
        month = input("월을 입력하세요(01~12월):")
        day = input("날짜를 입력하세요(01~12월):")
        date = year+'-'+month+'-'+day
        cursor = tweepy.Cursor(self.api.search_tweets, q=keyword, until=date, count=100, geocode=location, include_entities=True)

        wr.writerow(["number", "date", "user_id", "user_name", "user_screen_name", "tweet", "retweets", "likes", "follower_number", "following_number"])
        for i, tweet in enumerate(cursor.items()):
            wr.writerow([i, tweet.created_at, tweet.user.id, tweet.user.name, tweet.user.screen_name,tweet.text,tweet.retweet_count, tweet.favorite_count, tweet.user.followers_count, tweet.user.friends_count])
            
        return "twitter.csv file uploaded"

import pandas as pd 

token = pd.read_csv("/Users/kimjuwon/Github/Capstone_Data_Crawling/twitter_crawler/token.csv", encoding='utf-8')
token = list(token.loc[0])

coin_crawler = tweepy_coin_crawler(token[0], token[1], token[2],token[3])
coin_crawler.tweepy_connect()

location = "%s,%s,%s" % ("35.95", "-95.712891", "1000km")  # 검색기준(대한민국 중심) 좌표, 반지름  
keyword = "bitcoin OR crypto"  

print(coin_crawler.tweepy_crawler_with_7_days(location, keyword))