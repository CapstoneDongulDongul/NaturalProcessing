import tqdm
import pandas as pd

month_dict = {
        'Jan':'01',
        'Feb':'02',
        'Mar':'03',
        'Apr':'04',
        'May':'05',
        'Jun':'06',
        'Jul':'07',
        'Aug':'08',
        'Sep':'09',
        'Oct':'10',
        'Nov':'11',
        'Dec':'12',
    }
    
class coin_data_pipeline: 

    def __init__(self,coin_df, start_date = None, finish_date = None):
        self.coin_df = coin_df
        self.start_date = start_date
        self.finish_date = finish_date

    def convert_for_analysis(self):
        length = self.coin_df.shape[0]

        for i in tqdm.tqdm(range(length)):
            date=self.coin_df['date'].iloc[i]
            self.coin_df['date'].iloc[i] = date[8:]+'-'+month_dict[date[:3]]+'-'+date[4:6]
            self.coin_df['open'].iloc[i] = float(self.coin_df['open'].iloc[i][1:].replace(",", ""))
            self.coin_df['high'].iloc[i] = float(self.coin_df['high'].iloc[i][1:].replace(",", ""))
            self.coin_df['low'].iloc[i] = float(self.coin_df['low'].iloc[i][1:].replace(",", ""))
            self.coin_df['close'].iloc[i] = float(self.coin_df['close'].iloc[i][1:].replace(",", ""))
            self.coin_df['vol'].iloc[i]= float(self.coin_df['vol'].iloc[i][1:].replace(",", ""))
            self.coin_df['market cap'].iloc[i] = float(self.coin_df['market cap'].iloc[i][1:].replace(",", ""))

        return self.coin_df.sort_values('date',inplace=True)

    def look_up_data(self, start_date, finish_date): 
        if self.start_date == None : 
            self.start_date = start_date
        if self.finish_date == None :
            self.finish_date = finish_date
        length = self.coin_df.shape[0]

        check = False
        new_data_dict= {'date':[],'open':[],'high':[],'low':[],'close':[],'vol':[],'market cap':[]}
        for i in tqdm.tqdm(range(length)):
            if self.coin_df['date'].iloc[i] == start_date : 
                check = True 

            if check == True : 
                new_data_dict['date'].append(self.coin_df['date'].iloc[i])
                new_data_dict['open'].append(self.coin_df['open'].iloc[i])
                new_data_dict['high'].append(self.coin_df['high'].iloc[i])
                new_data_dict['low'].append(self.coin_df['low'].iloc[i])
                new_data_dict['close'].append(self.coin_df['close'].iloc[i])
                new_data_dict['vol'].append(self.coin_df['vol'].iloc[i])
                new_data_dict['market cap'].append(self.coin_df['market cap'].iloc[i])

            if self.coin_df['date'].iloc[i] == finish_date : 
                check = False

        return pd.DataFrame(new_data_dict).reset_index(drop=True)
                
        