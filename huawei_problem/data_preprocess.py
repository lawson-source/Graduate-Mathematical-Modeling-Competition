import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class Data_preprocess(object):
    def __init__(self,data):
        self.data=data
    def deal_categorical_variable(self,column,label):
        probability = (self.data.groupby(column)[[label]].sum() / self.data[column].size)[label] / (
                    self.data.groupby(column)[[column]].size() / self.data[column].size)
        return probability

    def struc_distance(self,X,Y,x,y):
        data_distance = ((self.data[x] - self.data[X]) ** 2 + (self.data[y] - self.data[Y]) ** 2) ** 0.5
        return data_distance

    def struc_height(self,h,a,ca):
        data_height= self.data[h] + self.data[ca] - self.data[a]
        return data_height





def main():
    data=pd.read_csv('./train_set/whole.csv',nrows=100000000,)
    data['RSRP_label']=(data['RSRP']>=-103).astype('int')
    DP=Data_preprocess(data)
    pbaty_CI=DP.deal_categorical_variable('Clutter Index', 'RSRP_label')
    pbaty_CCI=DP.deal_categorical_variable('Cell Clutter Index', 'RSRP_label')
    data['Clutter Index']=data['Clutter Index'].replace(pbaty_CI.index,pbaty_CI)
    data['Cell Clutter Index']=data['Cell Clutter Index'].replace(pbaty_CCI.index,pbaty_CCI)
    data['ds']=DP.struc_distance('X','Y','Cell X','Cell Y')
    data['hb']=DP.struc_height('Height','Cell Altitude','Altitude')
    Xdata=data[{'Clutter Index','Cell Clutter Index','ds','hb','Frequency Band','RS Power','Cell Building Height','Building Height'}]
    ydata=data['RSRP']
    for column in Xdata.columns:
       Xdata[column]=(Xdata[column]-Xdata[column].mean())/Xdata[column].std()
    ydata=(ydata+103)/10
    return Xdata,ydata





if __name__ == '__main__':
    main()

