import numpy as np

from torch.utils import data
from utils import utils

class DataSet(data.Dataset):
    def __init__(self, data, seq_len, pre_len):
        super().__init__()
        # data:[timeseq, nodes]
        data = data
        seq_len = seq_len
        pre_len = pre_len

        CityX, CityY = [], []
        for i in range(len(data) - seq_len - pre_len):
            a = data[i: i + seq_len + pre_len]
            CityX.append(a[0:seq_len])
            CityY.append(a[seq_len:seq_len+pre_len])
        
        self.mean = np.mean(CityX)
        self.maxnum = np.max(CityX)
        self.minnum = np.min(CityX)
        self.median = np.median(CityX)
        CityX = (CityX - self.minnum)/((self.maxnum - self.minnum)+1e-8)
        
        std = np.std(CityX)

        self.data, self.pre = np.asarray(CityX, dtype=np.float32), np.asarray(CityY, dtype=np.float32)

    def reverse(self, data):
        return data*(self.maxnum - self.minnum) + self.minnum
    
    def data_attr(self):
        return self.DataAttr
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        CityX = self.data[index]
        CityY = self.pre[index]
        return CityX, CityY
