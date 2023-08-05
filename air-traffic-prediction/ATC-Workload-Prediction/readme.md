# Air Traffic Controller Workload Level Prediction via Conformalized Dynamic Graph Evolution
Author: Yutian Pang, Arizona State University <br>
Email: yutian.pang@asu.edu

## Highlights
- We predict the air traffic controller (ATC) workload level from the flight traffic recordings. 
- We structure the flight traffic information as dynamic graphs, varying with airspace spatiotemporal layouts. 
- We adopt the evolving graph convolutional network (EvolveGCN) for this time-series dynamic graph classification task. 
- We use conformal prediction for the improved ground-truth label coverage and show higher accuracy than simple classification.  
- The workload experimental data is collected through real-world human ATC controllers' experiments, where both traffic conditions and human subjects are recorded.
- The graph learning model, EvolveGCN, shows outstanding prediction capabilities compared with other baselines. 

## EvolveGCN: Evolving Graph Convolutional Networks for Dynamics Graphs
The details of [EvolveGCN](https://arxiv.org/abs/1902.10191) can be found in the original paper. 

## Code Descriptions:
- [x] Raw Workload Data Parser
- [x] Training Code for EvolveGCN-O/EvolveGCN-H/GCN-Baseline
- [x] Prediction Label Post-Processing Conformal Prediction

## Dependency:
* dgl
* pandas
* numpy

## Setup Environment
### Linux with GPU support
```bash
conda create --name dgl python==3.9.0
conda activate dgl
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
conda install -c dglteam dgl-cuda10.2
conda install pandas
```
## To Train a Model
```bash
python train.py --raw-dir ./workload_data/ --processed-dir ./workload_data/processed --eval-class-id 0 --gpu 7
```

## Citation
```
@article{PANG2023102113,
title = {Air traffic controller workload level prediction using conformalized dynamical graph learning},
journal = {Advanced Engineering Informatics},
volume = {57},
pages = {102113},
year = {2023},
issn = {1474-0346},
doi = {https://doi.org/10.1016/j.aei.2023.102113},
url = {https://www.sciencedirect.com/science/article/pii/S1474034623002410},
author = {Yutian Pang and Jueming Hu and Christopher S. Lieber and Nancy J. Cooke and Yongming Liu},
keywords = {Air traffic management, Aviation human factors, Controller workload, Graph neural network}
}
```

## Reference
Model training code borrowed heavily from [IBM/EvolveGCN](https://github.com/IBM/EvolveGCN)  
