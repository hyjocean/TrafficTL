import argparse
import numpy as np
import pandas as pd


def DCRNN_init(args):
    args.cl_decay_steps = 1000
    args.max_diffusion_step = 2
    args.filter_type = 'laplacian'
    args.num_rnn_layers = 1
    args.rnn_units = args.hidden_size
    args.use_curriculum_learning = False
    return args