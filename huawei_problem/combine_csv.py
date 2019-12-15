import os
import pandas as pd
from multiprocessing import Pool

class Comebine(object):

    def __init__(self, File_dir):
        self.File_dir =File_dir
        self.Filelist=os.listdir(File_dir)
    def read_to_file_first(self):
        data = pd.read_csv(self.File_dir + '/' + self.Filelist[0], )
        data.to_csv(self.File_dir + '/whole.csv', index=False, encoding='utf-8')
    def read_to_file(self,i):
        data = pd.read_csv(self.File_dir + '/' + self.Filelist[i])
        data.to_csv(self.File_dir + '/whole.csv', mode='a+', index=False, header=False)
        print(i)
    def use_multi(self,worknum=4):
        msgs = [i for i in range(1, len(self.Filelist))]
        pool = Pool(worknum)
        pool.map(self.read_to_file, msgs)
        pool.close()
        pool.join()



def main ():
        dirc='./train_set'
        combine= Comebine(dirc)
        combine.read_to_file_first()
        combine.use_multi(worknum=8)



if __name__ == '__main__':
        main()

