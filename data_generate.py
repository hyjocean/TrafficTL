import numpy as np
from pathlib import Path


def data_gen():
    src_data = np.random.randint(0, 120, size = (30*288, 1324)) # size=day_num*day_item, nodes
    trg_data = np.random.randint(0, 120, size = (30*288, 413))

    cur_parent = Path(__file__).parent
    np.savez(cur_parent.joinpath('data', 'src_data.npz'), src_data)
    np.savez(cur_parent.joinpath('data', 'trg_data.npz'), trg_data)

if __name__ == '__main__':
    data_gen()