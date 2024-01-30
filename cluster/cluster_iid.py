import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader

from utils.utils import load_config, seed_set, device_set
from cluster.IID_losses import IID_loss


# data 
class Dataset(data.Dataset):
    def __init__(self, data, seq_len, t, epision=1e-8):
        super().__init__()
        
        # mean = np.mean(data, axis=0).reshape(1, -1)
        self.raw_data = data
        x = []
        y = []

        for i in range(len(data) - seq_len):
            a = data[i: i+seq_len]
            b = data[i+seq_len:i+seq_len+t]
            x.append(a)
            y.append(b)


        self.data_x = np.asarray(x[:-t])
        self.data_x_t = np.asarray(x[t:])
        self.data_y = np.asarray(y[:-t])
        self.max_num = np.max(x)
        self.min_num = np.min(x)
        self.data_x = (self.data_x - self.min_num) / (self.max_num - self.min_num + epision)
        self.data_x_t = (self.data_x_t - self.min_num) / (self.max_num - self.min_num + epision)
        
    def reverse(self, data):
        return data*(self.max_num - self.min_num) + self.min_num
    
    # def data_attr(self):
    #     return self.DataAttr

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        x = self.data_x[index]
        x_t_out = self.data_x_t[index]
        y = self.data_y[index]
        return x, x_t_out, y


# net
class GRUNet(nn.Module):
    def __init__(self, bz, nodes, seq_len, pre_len, bidirectional=2, num_layers=2, hidden_size=32, input_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.bz = bz
        self.nodes = nodes
        self.pre_len = pre_len
        self.seq_len = seq_len

        num_layers = num_layers
        bidirectional = bidirectional

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False, dropout=0.2, bidirectional=bool(bidirectional-1))
        # self.softmax = nn.Softmax(dim=1)
        # self.linear = nn.Linear(hidden_size, cluster_num)
        self.state = nn.Parameter(torch.empty(num_layers*bidirectional, bz*nodes, hidden_size), requires_grad=True)
        nn.init.xavier_normal_(self.state)
        # nn.init.xavier_normal_(self.linear.weight)
        self.line_batch = nn.Sequential(nn.Linear(hidden_size*(2 if self.gru.bidirectional else 1), hidden_size),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, input_size),
                                        nn.BatchNorm1d(input_size),
                                        nn.ReLU())
        self.linear = nn.Linear(seq_len, pre_len)
        self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()


    def forward(self, x):
        # x: [b, seq, node]
        x = x.permute(1,0,2).reshape(-1, self.bz*self.nodes).unsqueeze(2)

        # x: [seq, b*nodes, 1]
        input = x
        state = self.state
        # input = x.reshape(-1, self.gru.input_size)
        
        x, h_out = self.gru(x, state) # [seq, B*N, bi*h], [bi*numlayer, b*nodes, hidden_size]

        x = self.line_batch(x.reshape(-1, self.hidden_size*(2 if self.gru.bidirectional else 1))).reshape(self.seq_len, self.bz*self.nodes, -1) # [seq, BN,1]
        x = input + x 

        out = x.unsqueeze(-1).reshape(self.seq_len, self.bz, self.nodes) # [bz, nd, seq_len]
        out = self.sigmoid(self.linear(out.permute(1,2,0))) # [bz, nd, pre_len]


        return out, h_out #  [bz, nd, pre_len]

class Classifer(nn.Module):
    def __init__(self, cluster_num, config) -> None:
        super().__init__()
        hidden_size = config['cluster']['gru_model']['hidden_size']
        bidirectional = config['cluster']['gru_model']['bidirectional']
        num_layers = config['cluster']['gru_model']['num_layers']
        self.start_size = bidirectional*num_layers*hidden_size
        self.net = nn.Sequential(nn.Linear(self.start_size , self.start_size//4),
                                nn.BatchNorm1d(self.start_size//4),
                                nn.ReLU(),
                                nn.Linear(self.start_size//4, cluster_num),
                                nn.BatchNorm1d(cluster_num),
                                nn.ReLU())
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, h, h_t):
        # h: [bi*numlayer, bz*nodes, hidden_size]
        h = h.permute(1, 0, 2).reshape(-1, self.start_size)
        h_t = h_t.permute(1, 0, 2).reshape(-1, self.start_size)
        
        h_o = self.net(h)
        h_to = self.net(h_t)

        res_h = self.soft_max(h_o)
        res_h_to = self.soft_max(h_to)
        
        return res_h, res_h_to

def load_data(config):
    src_data = np.load(config['basic']['data_path'][config['basic']['src_city']])['speed']
    trg_data = np.load(config['basic']['data_path'][config['basic']['trg_city']], allow_pickle=True)['speed'][0*288:1*288]
    
    copy_cnt = src_data.shape[0] / trg_data.shape[0] + 1
    trg_trans_data = np.tile(trg_data, (int(copy_cnt), 1))
    mix_src_trg_dt = np.concatenate((src_data, trg_trans_data[:src_data.shape[0]]), axis=1)

    domain_label = np.asarray([0]*src_data.shape[1] + [1]*trg_data.shape[1])
    return mix_src_trg_dt, domain_label


def Pre_gru(dataloader, config):
    lr, bz, epochs, input_size, hidden_size, num_layers, bidirectional = config['cluster']['gru_model']['lr'],config['cluster']['gru_model']['bz'],\
                                                                        config['cluster']['gru_model']['epochs'],config['cluster']['gru_model']['input_size'],\
                                                                        config['cluster']['gru_model']['hidden_size'],config['cluster']['gru_model']['num_layers'],\
                                                                        config['cluster']['gru_model']['bidirectional']       
    logger, seq_len, pre_len, device = config['basic']['logger'], config['cluster']['seq_len'], config['cluster']['t'], config['basic']['device']
    
    data_nodes = dataloader.dataset.data_x.shape[2]
    gru_model = GRUNet(bz, data_nodes, seq_len, pre_len, bidirectional, num_layers, hidden_size, input_size).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=lr)

    # train
    gru_model.train()
    best_epoch = {'epoch_id': 0, 'epoch_loss': 0, 'best_model_stat': 0}
    for epoch_id in range(epochs):
        gru_model.train()
        batch_loss = 0.
        for batch_id, (x, _, y) in enumerate(dataloader):
            optimizer.zero_grad()
            if x.shape[0] != bz:
                continue

            x = torch.as_tensor(x, dtype=torch.float32).to(device)
            y = torch.as_tensor(y, dtype=torch.float32).to(device)
            x_out, _ = gru_model(x)
            
            x_out = dataloader.dataset.reverse(x_out)
            
            loss = criterion(x_out, y.permute(0,2,1))
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            if batch_id%(len(dataloader)//10) == 0:
                logger.info(f"Epoch_gru: {epoch_id}\tsrc_trg_cln: {config['basic']['src_city']}_{config['basic']['trg_city']}\tavg_loss:{batch_loss/(batch_id+1):.4f},")

        cur_epoch_loss = batch_loss / (batch_id+1) 
        logger.info(f"Epoch_gru_END: {epoch_id}\tsrc_trg_cln: {config['basic']['src_city']}_{config['basic']['trg_city']}\tepoch_loss:{cur_epoch_loss:.4f}\n")
        if cur_epoch_loss < best_epoch['epoch_loss'] or best_epoch['epoch_loss'] == 0.:
            best_epoch['epoch_id'] = epoch_id
            best_epoch['epoch_loss'] = cur_epoch_loss
            best_epoch['best_model_stat'] = gru_model.state_dict()
        if (best_epoch['epoch_id'] != 0 and epoch_id > best_epoch['epoch_id'] + config['cluster']['gru_model']['early_epoch_stop']) or (epoch_id == epochs-1):
            gru_model_pth=Path(__file__).parent.joinpath('model_pth','gru_model', f"{config['basic']['src_city']}_{config['basic']['trg_city']}.pth")
            if not gru_model_pth.parent.exists():
                gru_model_pth.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_epoch, gru_model_pth)
            break
        

    return gru_model


            
def Classify_layer(cluster_num_list, dataloader, gru_model, config):
    logger, device = config['basic']['logger'], config['basic']['device']
    epochs, lr, bz = config['cluster']['cls_model']['epochs'], config['cluster']['cls_model']['lr'], config['cluster']['gru_model']['bz'] 
    gru_model.eval()


    best_cluster = {'bst_num': 0, 'best_model_stat': 0, 'bst_loss': 0}
    improvement = 0.
    imp_eps = 0.1
    for cluster_num in cluster_num_list:
        logger.info(f"CLUSTER NUMBER: {cluster_num}")
        cls_model = Classifer(cluster_num, config).to(device)
        # cls_criterion = IID_loss()
        cls_optimizer = torch.optim.Adam(cls_model.parameters(), lr=lr)
        cls_model.train()
        for epoch_id in range(epochs):
            cls_model.train()
            best_epoch = {'epoch_id': 0, 'epoch_loss': 0, 'epoch_stat': 0}
            batch_loss = 0.
            for batch_id, (x, x_t, y) in enumerate(dataloader):
                cls_optimizer.zero_grad()
                if x.shape[0] != bz:
                    continue
                x = torch.as_tensor(x, dtype=torch.float32).to(device)
                x_t = torch.as_tensor(x, dtype=torch.float32).to(device)

                with torch.no_grad():
                    _, h_o = gru_model(x)
                    _, h_to = gru_model(x_t)

                res_h, res_ht = cls_model(h_o, h_to)

                loss, _ = IID_loss(res_h, res_ht)
                loss.backward()
                cls_optimizer.step()

                batch_loss += loss.item()
                if batch_id%(len(dataloader)//10) == 0:
                    logger.info(f"Epoch_cls: {epoch_id}_{cluster_num}\tsrc_trg_cln: {config['basic']['src_city']}_{config['basic']['trg_city']}_{cluster_num}\tavg_loss:{batch_loss/(batch_id+1):.4f},")

            cur_epoch_loss = batch_loss / (batch_id+1) 
            logger.info(f"Epoch_cls_END: {epoch_id}_{cluster_num}\tsrc_trg_cln: {config['basic']['src_city']}_{config['basic']['trg_city']}_{cluster_num}\tepoch_loss:{cur_epoch_loss:.4f}\n")
            if cur_epoch_loss < best_epoch['epoch_loss'] or best_epoch['epoch_loss'] == 0.:
                best_epoch['epoch_id'] = epoch_id
                best_epoch['epoch_loss'] = cur_epoch_loss
                best_epoch['epoch_stat'] = cls_model.state_dict()
            if (best_epoch['epoch_id'] != 0 and epoch_id > best_epoch['epoch_id'] + config['cluster']['cls_model']['early_epoch_stop']) or (epoch_id == epochs - 1):
                if best_cluster['bst_num'] == 0 or best_epoch['epoch_loss'] < best_cluster['bst_loss']:
                    if improvement == 0.:
                        improvement = - best_epoch['epoch_loss'] / cluster_num
                    else:
                        improvement = - (best_epoch['epoch_loss'] - best_cluster['bst_loss']) / (cluster_num - best_cluster['bst_num']) 
                    if improvement >= imp_eps:
                        best_cluster['bst_num'] = cluster_num
                        best_cluster['bst_loss'] = best_epoch['epoch_loss'] 
                        best_cluster['best_model_stat'] = best_epoch['epoch_stat']

                    logger.info(f"CLUSTER STATE: \t cur_cls_num:{cluster_num}\tbst_cls_num:{best_cluster['bst_num']}\tbst_iid_loss:{best_cluster['bst_loss']}\n\n\n")
                else: 
                    logger.info(f"CLUSTER STATE: \t cur_cls_num:{cluster_num}\tbst_cls_num:{best_cluster['bst_num']}\tbst_iid_loss:{best_cluster['bst_loss']}\n\n\n")
                    break
            
    logger.info(f"FINAL CLUSTER ENDS:\tbest_cls_num:{best_cluster['bst_num']}\tbest_iid_loss:{best_cluster['bst_loss']}\n")
    cls_model_pth=cls_model_pth=Path(__file__).parent.joinpath('model_pth','cls_model',f"{config['basic']['src_city']}_{config['basic']['trg_city']}_cln_{best_cluster['bst_num']}.pth")
    if not cls_model_pth.parent.exists():
        cls_model_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_cluster, cls_model_pth)

    return best_cluster




# model: gru + classify
def cluster_main(cluster_num_list, config):
    logger, device = config['basic']['logger'], config['basic']['device']

    seed = seed_set(config['basic']['SEED'])
    config['basic']['SEED'] = seed
    logger.info(f"MODEL SEED: {config['basic']['SEED']}")
    device_set(config['basic']['device'])
    
    seq_len, t, bz = config['cluster']['seq_len'], config['cluster']['t'], config['cluster']['gru_model']['bz']

    logger.info(f"LOADING DATA ...")
    data, domain_label = load_data(config)
    mix_dataset = Dataset(data, seq_len, t)
    dataloader = DataLoader(mix_dataset, batch_size=bz, shuffle=True, num_workers=8, pin_memory=True)

    if config['cluster']['gru_model']['path']:
        logger.info(f"LOADING gru_model ...")
        gru_msg = torch.load(config['cluster']['gru_model']['path'])
        gru_model_stat = gru_msg['best_model_stat']
        gru_model = GRUNet(bz, data.shape[1], seq_len, t, bidirectional=config['cluster']['gru_model']['bidirectional'],\
                                                num_layers=config['cluster']['gru_model']['num_layers'],\
                                                hidden_size=config['cluster']['gru_model']['hidden_size'],\
                                                input_size=config['cluster']['gru_model']['input_size']).to(device)
        gru_model.load_state_dict(gru_model_stat)
    else:
        logger.info(f"Train gru_model ...")
        gru_model = Pre_gru(dataloader, config)
    gru_model.eval()

    if config['cluster']['cls_model']['path']:
        logger.info(f"LOADING cls_model ...")
        cls_model_msg = torch.load(config['cluster']['cls_model']['path'])
        logger.info(f"cluster: {cls_model_msg['bst_num']}\tIID_Loss: {cls_model_msg['bst_loss']}")
    else:
        logger.info(f"Train cls_model ...")
        cls_model_msg = Classify_layer(cluster_num_list, dataloader, gru_model, config)
    
    cluster_num = cls_model_msg['bst_num']
    state_dict = cls_model_msg['best_model_stat']

    # reinit
    cls_model = Classifer(cluster_num, config).to(device)
    cls_model.load_state_dict(state_dict)
    cls_model.eval()

    nodes_cls = np.empty((data.shape[1], cluster_num))
    logger.info(f"CALCULATE THE FINAL RES ...")
    for batch_id, (x, x_t, _) in enumerate(tqdm(dataloader)):
        if x.shape[0] != bz:
            continue
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        x_t = torch.as_tensor(x, dtype=torch.float32).to(device)
            
        _, h_o = gru_model(x)
        _, h_to = gru_model(x_t)

        res_h, res_ht = cls_model(h_o, h_to)
        cls_h, cls_ht = np.argmax(res_h.detach().cpu().numpy(), axis=1), np.argmax(res_ht.detach().cpu().numpy(), axis=1)
        cls_h, cls_ht = np.transpose(cls_h.reshape(bz, -1),(1, 0)), np.transpose(cls_ht.reshape(bz, -1),(1, 0))
        nodes_cls += np.vstack([np.bincount(x, minlength=cluster_num) for x in cls_h])
        nodes_cls += np.vstack([np.bincount(x, minlength=cluster_num) for x in cls_ht])

    final_res = np.argmax(nodes_cls, axis=1) # [nodes]
    cur_path = Path(__file__).parent
    config['cluster']['results']['path'] = Path(f"{cur_path}/res/{config['basic']['src_city']}_{config['basic']['trg_city']}_clnum_{cluster_num}.npz")    
    if not config['cluster']['results']['path'].parent.exists():
        config['cluster']['results']['path'].parent.mkdir(parents=True, exist_ok=True)
    np.savez(config['cluster']['results']['path'], cluster_num=cluster_num, cluster_res = final_res, domain_label = domain_label)
    return cluster_num, final_res, domain_label, config['cluster']['results']['path']
