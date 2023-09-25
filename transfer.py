#coding=utf-8
from nis import cat
import os
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import torch
import torch.nn as nn
import numpy as np 
import pandas as pd
import argparse
import datetime
import random
import math
from pathlib import Path

from torch.utils.data import DataLoader
from utils.utils import seed_set, DTW_adj, device_set
from cluster import cluster
from data import data_process
from Model.model import get_model
from Model import encoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def evaluation(a,b):
    a = a.flatten()
    b = b.flatten()
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a,b)
    mape = mean_absolute_percentage_error(a,b)
    # F_norm = 1-(np.abs((a-b)/a).sum()/a.size)
    # r2 = 1 - ((a-b)**2).sum()/((a-a.mean())**2).sum()
    # var = 1-(np.var(a-b))/np.var(a)
    return mae, rmse, mape #, F_norm, r2, var


def train_epochs(config, cur_num, dataloader, adj, dm_mask):
    # get cluster data and adj
    logger, device, epochs = config['basic']['logger'], config['basic']['device'], config['transfer']['epochs']
    cluster_num = cur_num
    # init model
    device_set(device)
    model = get_model(config['transfer']['backbone']['name'], adj, config)
    epsilon = 1e-8
    optimizer = torch.optim.Adam(model.parameters(), lr=config['transfer']['lr'], eps=epsilon)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8, 10],
                                                            gamma=0.1)
    criterion = nn.L1Loss()
    
    best_epoch = {'epoch_id': 0, 'epoch_loss': 0, 'model': 0}
    for epoch_id in range(epochs):
        model.train()
        batch_loss = 0.
        for batch_id, (x_batch, y_batch) in enumerate(dataloader):
            if x_batch.shape[0] != config['transfer']['bz']:
                continue
            optimizer.zero_grad()
            train_x = x_batch.permute([0,2,1]).to(device)
            train_y = y_batch.permute([0,2,1]).to(device) # (batch, nodes, seq_len)
            
            res = model(train_x)
            res = dataloader.dataset.reverse(res.permute(1,2,0))

            loss = criterion(res, train_y)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            if batch_id%(len(dataloader)//10) == 0:
                logger.info(f"TRAIN_Epoch_transfer: {batch_id:03d}/{len(dataloader)}_{cluster_num}/{config['cluster']['results']['bs_num']}\tsrc_trg: {config['basic']['src_city']}_{config['basic']['trg_city']}\tavg_loss:{batch_loss/(batch_id+1):.4f},")
        
        cur_epoch_loss = batch_loss / (batch_id + 1)
        logger.info(f"TRAIN_Epoch_transfer_END: {epoch_id:03d}/{epochs}_{cluster_num}/{config['cluster']['results']['bs_num']}\tsrc_trg: {config['basic']['src_city']}_{config['basic']['trg_city']}\tavg_loss:{cur_epoch_loss:.4f}\n")
        if cur_epoch_loss < best_epoch['epoch_loss'] or best_epoch['epoch_loss'] == 0:
            best_epoch['epoch_id'] = epoch_id
            best_epoch['epoch_loss'] = cur_epoch_loss
            best_epoch['model'] = model

        if (best_epoch['epoch_id'] != 0 and epoch_id > best_epoch['epoch_id'] + config['transfer']['backbone']['early_epoch_stop']) or (epoch_id == epochs-1):
            logger.info(f"TRAIN_Epoch_transfer_BEST: {best_epoch['epoch_id']:03d}/{epochs}_{cluster_num}/{config['cluster']['results']['bs_num']}\tsrc_trg: {config['basic']['src_city']}_{config['basic']['trg_city']}\tavg_loss:{best_epoch['epoch_loss']:.4f}\n")
            trs_model_pth = Path(__file__).parent.joinpath('model_pth','trs_model', f"{config['basic']['src_city']}_{config['basic']['trg_city']}_{cluster_num}_{config['cluster']['results']['bs_num']}.pth")
            if not trs_model_pth.parent.exists():
                trs_model_pth.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_epoch['model'], trs_model_pth)
            break
    return model

def val_epochs(model, config, cur_num, dataloader, adj, dm_mask):
    logger, device, epochs = config['basic']['logger'], config['basic']['device'], config['transfer']['epochs']
    cluster_num = cur_num
    device_set(device)
    model.eval()
    criterion = nn.L1Loss()
    batch_mae, batch_rmse, batch_mape, batch_loss = 0. , 0. , 0. , 0.
    for batch_id, (x_batch, y_batch) in enumerate(dataloader):
        if x_batch.shape[0] != config['transfer']['bz']:
            continue
        val_x = x_batch.permute([0,2,1]).to(device)
        val_y = y_batch.permute([0,2,1]).to(device) # (batch, nodes, seq_len)
        with torch.no_grad():
            res = model(val_x)
        res = dataloader.dataset.reverse(res.permute(1,2,0))
        loss = criterion(res, val_y)

        mae, rmse, mape = evaluation(res[:,dm_mask,:].cpu().detach().numpy(), val_y[:, dm_mask, :].cpu().detach().numpy())
        batch_mae+=mae
        batch_rmse+=rmse
        batch_mape+=mape
        
        batch_loss += loss.item()
        if batch_id%(len(dataloader)//10) == 0:
            logger.info(f"VAL_batch_transfer: {batch_id:03d}/{len(dataloader)}_{cluster_num}/{config['cluster']['results']['bs_num']}\t"
                        f"src_trg_cln: {config['basic']['src_city']}_{config['basic']['trg_city']}\t"
                        f"avg_loss4cities: {batch_loss/(batch_id+1):.4f}\t"
                        f"avg_mae: {batch_mae/(batch_id+1):.4f}\t"
                        f"avg_rmse: {batch_rmse/(batch_id+1):.4f}\t"
                        f"avg_mape:{batch_mape/(batch_id+1):.4f},")
            
    logger.info(f"VAL_epoch_transfer_END: {cluster_num}/{config['cluster']['results']['bs_num']}\t"
                        f"src_trg_cln: {config['basic']['src_city']}_{config['basic']['trg_city']}\t"
                        f"avg_mae: {batch_mae/(batch_id+1):.4f}\t"
                        f"avg_rmse: {batch_rmse/(batch_id+1):.4f}\t"
                        f"avg_mape:{batch_mape/(batch_id+1):.4f},\n")
    

    return model, [batch_mae/(batch_id+1), batch_rmse/(batch_id+1), batch_mape/(batch_id+1)]
    

def main_transfer(config): 
    logger = config['basic']['logger']
    logger.info(config)
    logger.info(f"CUR SEED: {config['basic']['SEED']}")

    seed = seed_set(config['basic']['SEED'])
    config['basic']['SEED'] = seed
    logger.info(f"CUR update SEED: {config['basic']['SEED']}")

    raw_src_data = np.load(config['basic']['data_path'][config['basic']['src_city']], allow_pickle=True)['speed']
    raw_trg_data = np.load(config['basic']['data_path'][config['basic']['trg_city']], allow_pickle=True)['speed']
    src_data_train = raw_src_data[:config['transfer']['src_days4train']*config['basic']['items_day']]
    trg_data_train = raw_trg_data[:config['transfer']['trg_days4train']*config['basic']['items_day']]
    trg_data_val = raw_trg_data[config['transfer']['trg_days4train']*config['basic']['items_day']:config['transfer']['trg_days4val']*config['basic']['items_day']]
    trg_data_test = raw_trg_data[config['transfer']['trg_days4val']*config['basic']['items_day']:config['transfer']['trg_days4test']*config['basic']['items_day']]

    cl_res_msg = np.load(config['cluster']['results']['path'], allow_pickle=True)
    config['cluster']['results']['bs_num']=cl_res_msg['cluster_num']
    config['cluster']['results']['res']=cl_res_msg['cluster_res']
    config['cluster']['results']['domain']=cl_res_msg['domain_label']

    total_res = []
    dtw_data = np.hstack((src_data_train[:config['basic']['items_day']],trg_data_train[:config['basic']['items_day']]))
    for num in range(config['cluster']['results']['bs_num']):
        if not (config['cluster']['results']['res'] == num).any():
            continue
        else:
            logger.info(f"Now train cluster num == {num}/{config['cluster']['results']['bs_num']} ... ")
            cls_cfg = {}
            cls_mask = (config['cluster']['results']['res'] == num)
            dom_mask = config['cluster']['results']['domain'].astype(bool)
            cls_dom_msk = dom_mask[cls_mask]
            dtw_ndata = dtw_data[:,cls_mask]
            dtw_adj = DTW_adj(dtw_ndata)
            mean = dtw_adj.mean()
            dtw_adj[dtw_adj<mean] = 0
            dtw_adj[dtw_adj>mean] = 1
            cls_cfg['adj'] = dtw_adj
            logger.info(f"NUM: {num}/{config['cluster']['results']['bs_num']}, There are total {dtw_adj.shape[0]} nodes, {(~cls_dom_msk).sum()} for src, {cls_dom_msk.sum()} for trg")
            
            if cls_dom_msk.sum() == 0:
                logger.info(f"trg city not be assigned src data, enter the next cluster ... \n\n")
                continue
            
            model_path = Path(__file__).parent.joinpath('model_pth','trs_model', f"{config['basic']['src_city']}_{config['basic']['trg_city']}_{num}_{config['cluster']['results']['bs_num']}.pth")
            if not model_path.exists():
                logger.info(f"Training phase ...")
                assert src_data_train.shape[0] % trg_data_train.shape[0] == 0, 'train data shape not match'
                copy_num = src_data_train.shape[0] / trg_data_train.shape[0]
                train_data = np.hstack((src_data_train,np.tile(trg_data_train, (int(copy_num), 1))))
                train_dataset = data_process.DataSet(train_data[:, cls_mask], config['transfer']['seq_len'], config['transfer']['pre_len'])
                train_loader = DataLoader(train_dataset, batch_size=config['transfer']['bz'], shuffle=True, num_workers=16, pin_memory=True)
                train_model = train_epochs(config, num, train_loader, dtw_adj, cls_dom_msk)
            else:
                logger.info(f"LOADING trs_model ...")
                train_model = torch.load(model_path)
                # train_model = get_model(config['transfer']['backbone']['name'], dtw_adj, config)
                # train_model.load_state_dict(state, strict=False)
                
            
            logger.info(f"Validation phase ...")
            assert src_data_train.shape[0] % trg_data_val.shape[0] == 0, 'val data shape not match'
            copy_num = src_data_train.shape[0] / trg_data_val.shape[0]
            val_data = np.hstack((src_data_train,np.tile(trg_data_val, (int(copy_num), 1))))
            #TODO
            val_dataset = data_process.DataSet(val_data[:, cls_mask], config['transfer']['seq_len'], config['transfer']['pre_len'])
            val_loader = DataLoader(val_dataset, batch_size=config['transfer']['bz'], shuffle=False, num_workers=16, pin_memory=True)
            val_model, val_res = val_epochs(train_model, config, num, val_loader, dtw_adj, cls_dom_msk)


            logger.info(f"Testing phase ...")
            assert src_data_train.shape[0] % trg_data_test.shape[0] == 0, 'test data shape not match'
            copy_num = src_data_train.shape[0] / trg_data_test.shape[0]
            test_data = np.hstack((src_data_train,np.tile(trg_data_test, (int(copy_num), 1))))
            test_dataset = data_process.DataSet(test_data[:, cls_mask], config['transfer']['seq_len'], config['transfer']['pre_len'])
            test_loader = DataLoader(test_dataset, batch_size=config['transfer']['bz'], shuffle=False, num_workers=16, pin_memory=True)
            _, test_res = val_epochs(val_model, config, num, test_loader, dtw_adj, cls_dom_msk)

            total_res.append(test_res)
        logger.info(f"Cluster num == {num} training ending ...\n\n\n")


    logger.info(f"{total_res}")
    final_metric = np.mean(np.asarray(total_res), axis=0)
    logger.info(f"{final_metric}")
    logger.info(f"FINAL trg city metric: \tmae_{final_metric[0]:.4f}\trmse_{final_metric[1]:.4f}\tmape_{final_metric[2]:.4f}")
            


