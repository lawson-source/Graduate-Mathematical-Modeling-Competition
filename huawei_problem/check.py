import pandas as pd
class Check(object):
    def __init__(self,data):
        self.data=data

    def check_na(self):
        name_na={}
        total_na=self.data.isna().sum()
        for index in total_na.index:
           if total_na[index] !=0:
              name_na.update({index:total_na[index]})
        return name_na

    def drop_na(self):
        data=self.data.dropna()
        return data

    def replace_na(self,method='mean'):
        data=self.data.fillna(method=method)
        return data

def main():
    data=pd.read_csv('./train_set/whole.csv')
    check=Check(data)
    check_result=check.check_na()
    if len(check_result)==0:
        print('无空值')
    else:
        check.drop_na()





if __name__ == '__main__':
    main()