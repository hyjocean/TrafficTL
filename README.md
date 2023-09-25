# TrafficTL
The code for paper "[Traffic Prediction with Transfer Learning: A Mutual Information-based Approach](https://ieeexplore.ieee.org/abstract/document/10105852)"


<img width="929" alt="image" src="https://github.com/hyjocean/TrafficTL/assets/41313661/8b832609-2cbb-4682-acde-74d030edc85a">

# Notes
1. This version is a reconstruct code for TrafficTL, so it omits some complicated operations in papers like replace nodes in source city with nodes in target city.


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

