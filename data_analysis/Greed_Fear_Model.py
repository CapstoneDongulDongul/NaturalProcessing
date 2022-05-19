import pandas as pd
from sentimentalAnalysis import sentimental_analysis
import matplotlib.pyplot as plt
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from coincrawler import coin_crawl_his
from data_pipeline import coin_data_pipeline
from data_concat import data_concat
import statsmodels.api as sm 
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

class sentimental_score():
    def __init__(twitter_csv):
        self.twitter_csv = None
    
    def make_greed_fear_score():
        twitter_csv['Greed_Fear_Score']=twitter_csv['vader']*50 +twitter_csv['textblob']*50
        
    def show_monthly_score(self, year,month):
        count = 0
        
        if 'date' in twitter_csv.columns :
            if month < 10 : 
                month = '0'+str(month)
            year = str(year)
            check_monthly = year+'-'+month
        for i in range(len(self.twitter_csv)):
            if self.twitter_csv['date'].iloc[i][:8]==check_monthly:
                print(self.twitter_csv['Greed_fear_Score'].iloc[i])
        else : 
            print("this is not twitter_data")
        print("데이터 수: ",count)
            
class Greed_Fear_Model:
    def __init__(self,tweet_train=None, tweet_test=None):
        self.tweet_train = tweet_train
        self.tweet_test = tweet_test
        self.coin_data = None
        self.train_daily_score = None
        self.test_daily_score = None
        self.normed_train_data = None
        self.normed_test_data = None
        self.model = None
        
    # train_data, test_data set 결측치 제거 및 분석 도구
    def train_check_missing_value(self):
        return self.tweet_train.isna().sum()
        
    def train_fillna(self,f):
        return self.tweet_train.filna(f)
    
    def train_shape(self):
        return self.tweet_train.shape
    
    def train_dtypes(self):
        return self.tweet_train.dtypes
    
    def test_check_missing_value(self):
        return self.tweet_test.isna().sum()
    
    def test_fillna(self,f):
        return self.tweet_test.filna(f)
    
    def test_shape(self):
        return self.tweet_test.shape
        
    def coin_data_load(self,name):
        # data_load
        c_crawler = coin_crawl_his(name)
        c_crawler.startDriver()
        c_crawler.load_page_data()
        self.coin_data = c_crawler.get_data()
        c_crawler.quit_driver()
        # data_convert
        coin_converter = coin_data_pipeline(self.coin_data)
        self.coin_data = coin_converter.convert_for_analysis()
        # dtype지정
        self.coin_data = self.coin_data.astype({'open' : 'float64','close':'float64',
                                                'vol':'float64','low':'float64',
                                                'high':'float64','market cap':'float64'})
        return self.coin_data
    
    def convert_train_data(self,textblob_vader_train=None, flair_train=None):
        new_twitter_data = pd.merge(left = self.tweet_train , 
                                    right = textblob_vader_train[['vader','textblob']], 
                                    how = "inner", left_index = True, right_index=True)
        print(new_twitter_data.shape)
        new_twitter_data = pd.merge(left = new_twitter_data, right = flair_train["flair"], 
                                    left_index =True, right_index=True)
        print(new_twitter_data.shape)
        new_twitter_data = new_twitter_data.drop_duplicates()
        new_twitter_data = new_twitter_data.fillna(0)
        print(new_twitter_data.isna().sum())
        data_date = dict()
        for i in tqdm.notebook.tqdm(range(len(new_twitter_data))):
            row = new_twitter_data.iloc[i]
            key = str(new_twitter_data.iloc[i]['date'][:10])
            if len(row['date']) ==25:
                if len(str(row['follower_number'])) <= 11 and len(str(row['following_number'])) <= 11 and len(str(row['likes'])) <= 11:
                    if row['follower_number'] == 'False' : 
                        row['follower_number'] =0 
                    if row['following_number'] == 'False' : 
                        row['following_number'] = 0
                    if row['likes'] == 'False' : 
                        row['likes'] =0 
                    if key in data_date : 
                        data_date[key][0] += len(row['tweet']) 
                        data_date[key][1] += row['vader']
                        data_date[key][2] += row['textblob']
                        data_date[key][3] += 1
                        data_date[key][4] += int(float(row['follower_number']))
                        data_date[key][5] += int(float(row['following_number']))
                        data_date[key][6] += int(float(row['likes']))
                        data_date[key][7] += (row['flair'])
                    else : 
                        data_date[key] = [ len(row['tweet']), row['vader'], row['textblob'], 1, 
                                        int(float(row['follower_number'])),int(float(row['following_number'])),
                                        int(float(row['likes'])),row['flair'] ]
                        
                            
        data_datelist = []
        data_tweetlength = []
        data_date_vader_sum = []
        data_date_vader_avg = []
        data_date_textblob_sum = []
        data_date_textblob_avg = []
        data_date_count = []
        data_date_following_number_sum = []
        data_date_following_number_avg = []
        data_date_likes_sum = []
        data_date_likes_avg = []
        data_date_follower_number_sum = []
        data_date_follower_number_avg = []
        data_date_flair_sum = []
        data_date_flair_avg = []

        for key in tqdm.notebook.tqdm(data_date):
            data_datelist.append(key)
            data_tweetlength.append(data_date[key][0])
            data_date_vader_sum.append(data_date[key][1])
            data_date_vader_avg.append(data_date[key][1]/data_date[key][3])
            data_date_textblob_sum.append(data_date[key][2])
            data_date_textblob_avg.append(data_date[key][2]/data_date[key][3])
            data_date_flair_sum.append(data_date[key][7])
            data_date_flair_avg.append(data_date[key][7]/data_date[key][3])
            data_date_follower_number_sum.append(data_date[key][4])
            data_date_follower_number_avg.append(data_date[key][4]/data_date[key][3])
            data_date_following_number_sum.append(data_date[key][5])
            data_date_following_number_avg.append(data_date[key][5]/data_date[key][3])
            data_date_likes_sum.append(data_date[key][6])
            data_date_likes_avg.append(data_date[key][6]/data_date[key][3])
            data_date_count.append(data_date[key][3])
                
        sentimental_daily_score = pd.DataFrame({'date': data_datelist, 
                                        'tweet_length': data_tweetlength, 
                                        'vader_sum': data_date_vader_sum,
                                        'vader_avg':data_date_vader_avg,
                                        'textblob_sum': data_date_textblob_sum, 
                                        'textblob_avg':data_date_textblob_avg,
                                        'flair_sum': data_date_flair_sum,
                                        'flair_avg':data_date_flair_avg,
                                        'following_number_sum':data_date_following_number_sum,
                                        'following_number_avg':data_date_following_number_avg,
                                        'likes_sum':data_date_likes_sum,
                                        'likes_avg':data_date_likes_avg,
                                        'follower_number_sum':data_date_follower_number_sum,
                                        'follower_number_avg':data_date_follower_number_sum,
                                        'count':data_date_count
                                       })
        sentimental_daily_score_with_price = pd.merge(left = sentimental_daily_score , right = self.coin_data, 
                                              how = "inner", on = ["date"])
        target_date =[]
        length = sentimental_daily_score_with_price.shape[0]
        for i in tqdm.notebook.tqdm(range(length)):
            date=sentimental_daily_score_with_price.iloc[i]['date']
            year = int(date[:4])
            month = int(date[5:7])
            day = int(date[8:])+1
            if day >= 29 : 
                if month == 2 : 
                    if ((year%4 == 0 and year%100 != 0) or year%400 == 0):
                        day-=29
                        month+=1
                    else : 
                        day-=28
                        month+=1
                elif month in [1,3,5,7,8,10] and day>= 32: 
                    month+=1
                    day-=31
                elif month in [4,6,9,11] and day>=31:
                    month+=1
                    day-=30
                elif month ==12 and day>= 32: 
                    month=1
                    day-=31
            if day < 10 :
                day = '0'+str(day)
            if month < 10 : 
                month = '0'+str(month)

            new_date = str(year)+'-'+str(month)+'-'+str(day)
            target_date.append(new_date)
        sentimental_daily_score_with_price['target_date'] = target_date
            
        Coin_Target_Price = self.coin_data.rename(columns = {'date':'target_date',
                                                      'close' : 'target_close', 'open':'target_open',
                                                      'high':'target_high','low':'target_low',
                                                      'vol':'target_vol', 'market cap':'target_market cap'})
        sentimental_daily_score_with_price = pd.merge(left = sentimental_daily_score_with_price , 
                                                          right = Coin_Target_Price, 
                                                          how = "inner", 
                                                          on = ["target_date"])
        self.train_daily_score = sentimental_daily_score_with_price
        return self.train_daily_score
        
    def convert_test_data(self,textblob_vader_test=None, flair_test=None):
         
        new_twitter_data = pd.merge(left = self.tweet_test , 
                                    right = textblob_vader_test[['vader','textblob']], 
                                    how = "inner", left_index = True, right_index=True)
        new_twitter_data = pd.merge(left = new_twitter_data, right = flair_test["flair"], 
                                    left_index =True, right_index=True)
        new_twitter_data = new_twitter_data.drop_duplicates()
        new_twitter_data = new_twitter_data.fillna(0)
        print(new_twitter_data.isna().sum())
        data_date = dict()
        for i in tqdm.notebook.tqdm(range(len(new_twitter_data))):
            row = new_twitter_data.iloc[i]
            key = str(new_twitter_data.iloc[i]['date'][:10])
            if len(row['date']) ==25:
                if len(str(row['follower_number'])) <= 11 and len(str(row['following_number'])) <= 11 and len(str(row['likes'])) <= 11:
                    if row['follower_number'] == 'False' : 
                        row['follower_number'] =0 
                    if row['following_number'] == 'False' : 
                        row['following_number'] = 0
                    if row['likes'] == 'False' : 
                        row['likes'] =0 
                    if key in data_date : 
                        data_date[key][0] += len(row['tweet']) 
                        data_date[key][1] += row['vader']
                        data_date[key][2] += row['textblob']
                        data_date[key][3] += 1
                        data_date[key][4] += int(float(row['follower_number']))
                        data_date[key][5] += int(float(row['following_number']))
                        data_date[key][6] += int(float(row['likes']))
                        data_date[key][7] += (row['flair'])
                    else : 
                        data_date[key] = [ len(row['tweet']), row['vader'], row['textblob'], 1, 
                                          int(float(row['follower_number'])),int(float(row['following_number'])),
                                          int(float(row['likes'])),row['flair'] ]
                            
        data_datelist = []
        data_tweetlength = []
        data_date_vader_sum = []
        data_date_vader_avg = []
        data_date_textblob_sum = []
        data_date_textblob_avg = []
        data_date_count = []
        data_date_following_number_sum = []
        data_date_following_number_avg = []
        data_date_likes_sum = []
        data_date_likes_avg = []
        data_date_follower_number_sum = []
        data_date_follower_number_avg = []
        data_date_flair_sum = []
        data_date_flair_avg = []

        for key in tqdm.notebook.tqdm(data_date):
            data_datelist.append(key)
            data_tweetlength.append(data_date[key][0])
            data_date_vader_sum.append(data_date[key][1])
            data_date_vader_avg.append(data_date[key][1]/data_date[key][3])
            data_date_textblob_sum.append(data_date[key][2])
            data_date_textblob_avg.append(data_date[key][2]/data_date[key][3])
            data_date_flair_sum.append(data_date[key][7])
            data_date_flair_avg.append(data_date[key][7]/data_date[key][3])
            data_date_follower_number_sum.append(data_date[key][4])
            data_date_follower_number_avg.append(data_date[key][4]/data_date[key][3])
            data_date_following_number_sum.append(data_date[key][5])
            data_date_following_number_avg.append(data_date[key][5]/data_date[key][3])
            data_date_likes_sum.append(data_date[key][6])
            data_date_likes_avg.append(data_date[key][6]/data_date[key][3])
            data_date_count.append(data_date[key][3])
                
        sentimental_daily_score = pd.DataFrame({'date': data_datelist, 
                                    'tweet_length': data_tweetlength, 
                                    'vader_sum': data_date_vader_sum,
                                    'vader_avg':data_date_vader_avg,
                                    'textblob_sum': data_date_textblob_sum, 
                                    'textblob_avg':data_date_textblob_avg,
                                    'flair_sum': data_date_flair_sum,
                                    'flair_avg':data_date_flair_avg,
                                    'following_number_sum':data_date_following_number_sum,
                                    'following_number_avg':data_date_following_number_avg,
                                    'likes_sum':data_date_likes_sum,
                                    'likes_avg':data_date_likes_avg,
                                    'follower_number_sum':data_date_follower_number_sum,
                                    'follower_number_avg':data_date_follower_number_sum,
                                    'count':data_date_count
                                   })
        sentimental_daily_score_with_price = pd.merge(left = sentimental_daily_score , right = self.coin_data, 
                                          how = "inner", on = ["date"])
        target_date =[]
        length = sentimental_daily_score_with_price.shape[0]
        for i in tqdm.notebook.tqdm(range(length)):
            date=sentimental_daily_score_with_price.iloc[i]['date']
            year = int(date[:4])
            month = int(date[5:7])
            day = int(date[8:])+1
            if day >= 29 : 
                if month == 2 : 
                    if ((year%4 == 0 and year%100 != 0) or year%400 == 0):
                        day-=29
                        month+=1
                    else : 
                        day-=28
                        month+=1
                elif month in [1,3,5,7,8,10] and day>= 32: 
                    month+=1
                    day-=31
                elif month in [4,6,9,11] and day>=31:
                    month+=1
                    day-=30
                elif month ==12 and day>= 32: 
                        month=1
                        day-=31
            if day < 10 :
                day = '0'+str(day)
            if month < 10 : 
                month = '0'+str(month)

            new_date = str(year)+'-'+str(month)+'-'+str(day)
            target_date.append(new_date)
        sentimental_daily_score_with_price['target_date'] = target_date
            
        Coin_Target_Price = self.coin_data.rename(columns = {'date':'target_date',
                                                      'close' : 'target_close', 'open':'target_open',
                                                      'high':'target_high','low':'target_low',
                                                      'vol':'target_vol', 'market cap':'target_market cap'})
        sentimental_daily_score_with_price = pd.merge(left = sentimental_daily_score_with_price , 
                                                          right = Coin_Target_Price, 
                                                          how = "inner", 
                                                          on = ["target_date"])
        self.test_daily_score = sentimental_daily_score_with_price
        return self.test_daily_score
        
    def check_corr_with_sentimental_score(self):
        corr = self.train_daily_score.corr()
        return corr[['vader_sum', 'vader_avg','textblob_sum', 
              'textblob_avg','flair_sum','flair_avg']].loc[['target_open','target_close','target_high',
                                                            'target_low','target_vol','target_market cap']]
        
    def heatmap(self):
        sns.heatmap(self.train_daily_score.corr(), linewidths=.5, cmap = 'YlGnBu', annot=True)
        
    def train_normalize(self):
        normed = (self.train_daily_score.drop(['target_date','date'],axis=1)- self.train_daily_score.drop(['target_date','date'],axis=1).mean())/self.train_daily_score.drop(['target_date','date'],axis=1).std()
        normed = pd.merge(normed,self.train_daily_score['date'], 
                          left_index = True, right_index=True)
        self.normed_train_data = normed
        return self.normed_train_data
    
    def test_normalize(self):
        normed = (self.test_daily_score.drop(['target_date','date'],axis=1)- self.test_daily_score.drop(['target_date','date'],axis=1).mean())/self.test_daily_score.drop(['target_date','date'],axis=1).std()
        normed = pd.merge(normed,self.test_daily_score['date'], 
                          left_index = True, right_index=True)
        self.normed_test_data = normed
        return self.normed_test_data
    
    def make_model(self):
        twitter_dataset = self.normed_train_data
        y_target = twitter_dataset['target_close']
        x_data= twitter_dataset.drop(['target_close','date','target_open','target_high','target_low',
                              'target_vol','follower_number_avg','follower_number_sum','target_market cap',
                             'tweet_length','count','market cap','likes_avg','open','low','high',
                             'textblob_sum','vol','following_number_avg' ,'likes_sum','flair_sum','vader_sum'
                             ,'textblob_avg','following_number_sum'],axis=1)
        train_x, test_x, train_y, test_y = train_test_split(x_data, y_target, train_size=0.8, 
                                                            test_size=0.2,random_state = 2)
        test_x=sm.add_constant(test_x,has_constant='add')
        train_x=sm.add_constant(train_x,has_constant='add')
        self.model = sm.OLS(train_y, train_x)
        self.model = self.model.fit()
        self.VIF(x_data)
            
    def summary(self):
        self.model.summary()
        
    def VIF(self,x_data):
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(
        x_data.values, i) for i in range(x_data.shape[1])]
        vif["features"] = x_data.columns
        return vif
    
    def predict(self):
        test_dataset = self.normed_test_data
        test_y = test_dataset['target_close']
        test_x= test_dataset.drop(['target_close','date','target_open','target_high','target_low',
                              'target_vol','follower_number_avg','follower_number_sum','target_market cap',
                             'tweet_length','count','market cap','likes_avg','open','low','high',
                             'textblob_sum','vol','following_number_avg' ,'likes_sum','flair_sum','vader_sum'
                             ,'textblob_avg','following_number_sum'],axis=1)
        test_x=sm.add_constant(test_x,has_constant='add')
        
        predict = self.model.predict(test_x)*self.test_daily_score['target_close'].std()+self.test_daily_score['target_close'].mean()
        result = pd.DataFrame(predict, columns=['target_close'])
        result['predict_close']=test_y*self.test_daily_score['target_close'].std()+self.test_daily_score['target_close'].mean()
        result = pd.merge(left = result , right = self.test_daily_score[['date','vol','close','target_date']], 
                              left_index=True, right_index=True)
        result['Greed_Fear_Score']=(result['predict_close']-result['close'])/result['close']*1000
        result['difference']=result['target_close']-result['close']
        result['correct'] = pd.Series()
        for i in range(len(result)):
            if result['Greed_Fear_Score'].iloc[i] * result['difference'].iloc[i] > 0:
                result['correct'].iloc[i] =True
            else : 
                result['correct'].iloc[i] =False
        self.result = result
        return result
    
    def visualize(self):
        plt.figure(figsize=(20,8))
        plt.plot(self.result['predict_close'],label="predict price", color='red')
        plt.plot(self.result['target_close'],label="coin price", color='blue')
        plt.xticks(self.result.index,result['target_date'])
        plt.legend()
        plt.show()
    
    def save_csv(self,file_name):
        self.result[['date','predict_close','vol','close','Greed_Fear_Score','target_date']].to_csv(file_name+".csv")