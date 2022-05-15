import pandas as pd

class data_concat : 
    def __init__(self):
        self.result = None

    def concat(self, files):
        self.result=files[0]
        for i in range(1,len(files)):
            self.result = pd.concat([self.result, files[i]], axis=0)
        self.result.drop_duplicates()
        return self.result

    def save_csv(self, file_name):
        self.result.to_csv(file_name+'.csv',mode = 'w') 