import pandas as pd
import numpy as np 
import torch 
import torch.nn as nn
import random
from utils import utils 

class encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.init_state = nn.Parameter(torch.zeros(3, args.batch_size, args.hidden_size))
        # self.encoder = nn.ModuleList()
        # for i in range(args.seq_len):
        #     self.encoder.append(nn.GRU(args.default_nodes, hidden_size=args.hidden_size))
        self.encoder=nn.GRU(args.default_nodes, hidden_size=args.hidden_size,num_layers=3)
        # self.relu = [nn.ReLU(), nn.ReLU(), nn.ReLU()]
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)
        self.fc = nn.Linear(args.hidden_size, args.default_nodes)
        torch.nn.init.xavier_normal_(self.fc.weight)
        # self.fc2 = nn.Linear(args.seq_len, args.train_cluster)
        self.softmax = nn.Softmax(dim=1)
        self.device = args.device
        self.bn1 = nn.BatchNorm1d(args.default_nodes)

    def forward(self, x, state=None):
        """
        x: (b, seq, nodes)
        """
        x_norm = utils.norm_data(x.cpu().permute(0, 2, 1).reshape(-1, x.shape[1]))
        norm_train_x = torch.tensor(x_norm.minmax_data().reshape(x.shape)).permute(1, 0, 2).cuda(self.device)
        
        if state is None:
            state = self.init_state.to(norm_train_x.device)
        else:
            state = state.to(norm_train_x.device)

        # out = []
        # for i, layer in enumerate(self.encoder):
        #     state = layer(x[i], state)
        #     state = self.drop(state)
        #     out.append(state)

        out, hn = self.encoder(norm_train_x, state)

        # out = np.asarray(out).permute(1,0,2).reshape(-1, state.shape[1]) #64, 12，64

        label = self.relu(self.fc(hn).permute(1,2,0).reshape(-1, 3)) # (3, b, hidden) -> (3, b, df_nodes)
        return label, hn # (b*n, 3)


# def classify_coder(x, label, encoder, args):
#     """[summary]

#     Args:
#         x ([batch, seq, all_nodes]): [每一迭代下输入的数据]
#         label ([1, fake_lb]): [由dtw生成的fake_label]
#     """

#     itm_loss = 0
#     optimizer.zero_grad()

#     for i in range(x.shape[0]/df_nodes):
#         rand_sample_ids = random.sample(range(x.shape[0]), df_nodes)
#         x_input = x[:, :, random_sample_ids]
#         y_label = label[:, random_sample_ids].expand(batch, -1)
#         out = encoder(x_input)

#         loss = criterion(out, y_label)
#         loss.backward()
#         itm_loss += loss.item()

#     optimizer.step()
#     return encoder, itm_loss




# x = torch.randn(size=(64, 12, 100))
# net = Encoder(100, 64, 64, 12)
# label = net(x)