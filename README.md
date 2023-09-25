# TrafficTL
The code for paper "[Traffic Prediction with Transfer Learning: A Mutual Information-based Approach](https://ieeexplore.ieee.org/abstract/document/10105852)"


<img width="929" alt="image" src="https://github.com/hyjocean/TrafficTL/assets/41313661/8b832609-2cbb-4682-acde-74d030edc85a">

# Notes
1. This version is a reconstruct code for TrafficTL, so it omits some complicated operations in papers like replace nodes in source city with nodes in target city.


# Requirements
include but not limited
```python
torch
tqdm
dtaidistance
logging
pyyaml
numpy
sklearn
pathlib
```

# Step 1
For data privacy, please apply follow line to generate sample data for going through whole process.
```python
python data_generate.py
```

# Step 2
if you want to go through the pipeline, please use next command.
```python
python main.py --src_city 'src' --trg_city 'trg' --device 0
```
if you want to use it on your own data, please place your data on the file 'data' and specify data path in 'config.yaml'.
```python
python main.py --src_city 'xxx' --trg_city 'xxx' --device 0
```
please use city name replace 'xxx'.


# Thanks
Thanks for repository codes [IIC](https://github.com/xu-ji/IIC) and [DCRNN](https://github.com/chnsh/DCRNN_PyTorch)

# Citation
```python
@ARTICLE{10105852,
  author={Huang, Yunjie and Song, Xiaozhuang and Zhu, Yuanshao and Zhang, Shiyao and Yu, James J. Q.},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Traffic Prediction With Transfer Learning: A Mutual Information-Based Approach}, 
  year={2023},
  volume={24},
  number={8},
  pages={8236-8252},
  doi={10.1109/TITS.2023.3266398}}
```
