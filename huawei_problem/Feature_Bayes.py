import pandas as pd
class Feature_Bayes(object):
    def __init__(self,df_column,df_label):
        self.data=pd.DataFrame({'column':df_column,'label':df_label})
    def fea_stu_bayes(self):
        probability=(self.data.groupby('column')[['label']].sum()/self.data['column'].size)['label']/(self.data.groupby('column')[['column']].size()/self.data['column'].size)
        self.data['column']=self.data['column'].replace(probability.index,probability)
        return self.data['column']
def main():
        data = pd.read_csv('./train_set/whole.csv', nrows=100000)
        data['RSRP'] = (data['RSRP'] >= -103).astype('int')
        fb=Feature_Bayes(data['Clutter Index'],data['RSRP'])
        print(fb.fea_stu_bayes())
if __name__ == '__main__':
    main()
