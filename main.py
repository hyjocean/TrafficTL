
import datetime
import logging
import argparse

import torch
from pathlib import Path

# from torch.backends import cudnn
from utils.utils import configure_logger, load_config
from cluster.cluster_iid import cluster_main
from transfer import main_transfer

def get_args():

    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--src_city', type=str, default='src', help='The city be as a source data for transfer')
    parser.add_argument('--trg_city', type=str, default='trg', help='The city needed to be trasferred')
    parser.add_argument('--device', type=int, default=1, help='specify gpu')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # para set
    config_path = "config.yaml"
    config = load_config(config_path)
    num_cluster = config['cluster']['num_list']
    
    args = get_args()
    config['basic']['src_city'] = args.src_city
    config['basic']['trg_city'] = args.trg_city
    config['basic']['device'] = torch.device('cuda:'+str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # log设置
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    file_path = Path(__file__).parent
    config['log_file'] = Path(f"{file_path}/log/log_{time_str}.log")
    configure_logger(config['log_file'])
    logger = logging.getLogger()
    config['basic']['logger'] = logger

    logger.info(f"src_city: {config['basic']['src_city']}\ttrg_city: {config['basic']['trg_city']}\tdevice: {config['basic']['device']}\t")
    # cluster_process
    if not config['cluster']['results']['path']:
        cluster_num, final_res, domain_label, pth = cluster_main(num_cluster, config)
        config['cluster']['results']['path']=Path(pth)
        config['cluster']['results']['bs_num']=cluster_num
        config['cluster']['results']['res']=final_res
        config['cluster']['results']['domain']=domain_label
    # else:
    #     cl_res_msg = np.load(config['cluster']['results']['path'], allow_pickle=True)
    #     config['cluster']['results']['bs_num']=cl_res_msg['cluster_num']
    #     config['cluster']['results']['res']=cl_res_msg['cluster_res']
    #     config['cluster']['results']['domain']=cl_res_msg['domain_label']
    
    logger.info(f"START TRAINING IN BATCHES ...")
    # trainning
    
    main_transfer(config)


