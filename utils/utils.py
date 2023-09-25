import numpy as np 
import pandas as pd 
import yaml

import logging
import os
import torch
from sklearn.preprocessing import MinMaxScaler
from dtaidistance import dtw, clustering



def DTW_adj(data):
    data = data.T
    max_num = np.max(data, axis=1).reshape(-1, 1)
    min_num = np.min(data, axis=1).reshape(-1, 1)
    data = (data - min_num) /(max_num - min_num + 1e-6)
    series = np.matrix(data)
    ds = dtw.distance_matrix_fast(series)

    return ds

def device_set(device):
    # torch.cuda.set_device(device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True




def seed_set(SEED):
    if not SEED:
        SEED = np.random.randint(0, 10000)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return SEED 

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def dir_exist(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def read_data(file_path):
    # print('##### READ DATA #####')
    assert os.path.exists(file_path) == True
    feature = np.asarray(pd.read_csv(file_path, header=None).values)
    return feature

def get_label(label_dic:dict, label:list):

    i = 0
    for key in label_dic.keys():
        idx = list(label_dic[key])
        label[:, idx] = i
        i += 1
    return label

class norm_data():
    def __init__(self, data, axis=1):
        self.data = np.asarray(data)
        self.min_dim = np.min(self.data, axis=axis).reshape(-1, 1)
        self.max_dim = np.max(self.data, axis=axis).reshape(-1, 1)

    def minmax_data(self):
        # assert (self.max_dim - self.min_dim)==0
        dt = (self.data - self.min_dim) / ((self.max_dim - self.min_dim)+0.0001)
        return dt

    def get_static_attr(self):
        return {'min':self.min_dim, 'max':self.max_dim}

    def inverse(self, data):
        return data*(self.max_dim - self.min_dim) + self.min_dim


def configure_logger(log_file, log_level=logging.INFO):
    # 创建一个日志记录器
    logger = logging.getLogger()
    logging.basicConfig(level=log_level,  # 设置日志级别为INFO
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[
                        logging.FileHandler(log_file),  # 将日志写入文件
                        logging.StreamHandler()  # 将日志打印到控制台
                        ])

    # 创建一个文件处理程序
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setLevel(logging.INFO)

    # 创建一个格式化器
    # formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    # file_handler.setFormatter(formatter)

    # 添加处理程序到日志记录器
    # logger.addHandler(file_handler)

    # return logger

if __name__ == "__main__":
    configure_logger("my_log.log")
