import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Model import para_set
from Model.DCRNN import DCRNN


def get_model(model_name, adj, config):

    if model_name == 'dcrnn':
        # config = para_set.DCRNN_init(config)
        model = DCRNN.DCRNNModel(adj, config).cuda(config['basic']['device'])
    if model_name == 'MeST':
        # args = para_set.DCRNN_init(args)
        model = DCRNN.MetaST(adj, config).cuda(config['basic']['device'])
    if model_name == 'regiontrans':
        # args = para_set.DCRNN_init(args)
        model = DCRNN.RegionTrans(adj, config).cuda(config['basic']['device'])
    # if args.load_weights is not None:
    #     print('loading pretrained weights..')
    #     model.load_state_dict(torch.load(args.output_path+'/'+'model_weights'))

    return model
