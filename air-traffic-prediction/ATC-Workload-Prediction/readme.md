# Air Traffic Controller Workload Level Prediction via Conformalized Dynamic Graph Evolution
Author: Yutian Pang, Arizona State University <br>
Email: yutian.pang@asu.edu

## EvolveGCN: Evolving Graph Convolutional Networks for Dynamics Graphs
The details of [EvolveGCN](https://arxiv.org/abs/1902.10191) can be found in the original paper. 

## Code Descriptions:
- [ ] Raw Workload Data Parser
- [ ] Training Code for EvolveGCN-O/EvolveGCN-H/GCN-Baseline
- [ ] Prediction Label Post-Processing Conformal Prediction

## Dependency:
* dgl
* pandas
* numpy

## Setup Environment on Linux 
### Torch with GPU support
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
Our paper is under review. The code will be released once the paper is pending online. 

## Reference
Model training code borrowed heavily from [IBM/EvolveGCN](https://github.com/IBM/EvolveGCN)  
