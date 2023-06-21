# setup torch+dgl environment with cuda
conda create --name dgl python==3.9.0
conda activate dgl
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
conda install -c dglteam dgl-cuda10.2
conda install pandas

# run the data parser to get the graph featurs and graph labels
python /~/HDD1/ypang6/workload/DynamicGraph/workload_data/WorkloadParserASU.py

# train a model
Baseline:
python /~/HDD1/ypang6/workload/DynamicGraph/EvolveGCN-DGL-master/train.py --raw-dir /~/HDD1/ypang6/workload/DynamicGraph/workload_data/processed/baseline --processed-dir /~/HDD1/ypang6/workload/DynamicGraph/workload_data/processed/baseline --model EvolveGCN-O --gpu 3 

High Workload Nominal:
python /~/HDD1/ypang6/workload/DynamicGraph/EvolveGCN-DGL-master/train.py --raw-dir /~/HDD1/ypang6/workload/DynamicGraph/workload_data/processed/hi_wkld_nom --processed-dir /~/HDD1/ypang6/workload/DynamicGraph/workload_data/processed/hi_wkld_nom --model EvolveGCN-O --gpu 7 

High Workload Off-Nominal:
```
(dgl) paralab@paralab-gpu:~$ python /~/HDD1/ypang6/workload/DynamicGraph/EvolveGCN-DGL-master/train.py --raw-dir /~/HDD1/ypang6/workload/DynamicGraph/workload_data/processed/hi_wkld_off --processed-dir /~/HDD1/ypang6/workload/DynamicGraph/workload_data/processed/hi_wkld_off --model EvolveGCN-O --gpu 1
The preprocessed data already exists, skip the preprocess stage!
Epoch 0  >>>>>>>>>>>>>
Train | microF1:0.3095 | macroF1: 0.2460| Trainloss:1.94549918
Valid | microF1:0.6370 | macroF1: 0.3866 | Validloss:1.88206100
###################Epoch 0 Test###################
Test | Epoch 0 | microF1:0.6630 | macroF1: 0.3319
Epoch 1  >>>>>>>>>>>>>
Train | microF1:0.5299 | macroF1: 0.4182| Trainloss:2.08705568
Valid | microF1:0.6333 | macroF1: 0.3419 | Validloss:2.03887010
Epoch 2  >>>>>>>>>>>>>
Train | microF1:0.5985 | macroF1: 0.4605| Trainloss:2.14889717
Valid | microF1:0.6333 | macroF1: 0.3417 | Validloss:2.09834385
Epoch 3  >>>>>>>>>>>>>
Train | microF1:0.6394 | macroF1: 0.4955| Trainloss:2.15023088
Valid | microF1:0.6722 | macroF1: 0.3703 | Validloss:2.14165974
###################Epoch 3 Test###################
Test | Epoch 3 | microF1:0.6648 | macroF1: 0.3218
Epoch 4  >>>>>>>>>>>>>
Train | microF1:0.6117 | macroF1: 0.4598| Trainloss:2.10138750
Valid | microF1:0.6722 | macroF1: 0.3761 | Validloss:2.14943171
Epoch 5  >>>>>>>>>>>>>
Train | microF1:0.6350 | macroF1: 0.4794| Trainloss:2.12644243
Valid | microF1:0.7093 | macroF1: 0.4014 | Validloss:2.15673614
###################Epoch 5 Test###################
Test | Epoch 5 | microF1:0.6528 | macroF1: 0.3304
Epoch 6  >>>>>>>>>>>>>
Train | microF1:0.6321 | macroF1: 0.4739| Trainloss:2.14900088
Valid | microF1:0.7074 | macroF1: 0.4000 | Validloss:2.16025972
Epoch 7  >>>>>>>>>>>>>
Train | microF1:0.6394 | macroF1: 0.4787| Trainloss:2.14334750
Valid | microF1:0.7074 | macroF1: 0.4000 | Validloss:2.16103649
Epoch 8  >>>>>>>>>>>>>
Train | microF1:0.6277 | macroF1: 0.4653| Trainloss:2.14146137
Valid | microF1:0.7074 | macroF1: 0.4000 | Validloss:2.16161728
Epoch 9  >>>>>>>>>>>>>
Train | microF1:0.6219 | macroF1: 0.4581| Trainloss:2.10922122
Valid | microF1:0.7074 | macroF1: 0.4000 | Validloss:2.16257763
Epoch 10  >>>>>>>>>>>>>
Train | microF1:0.6234 | macroF1: 0.4557| Trainloss:2.12758493
Valid | microF1:0.7074 | macroF1: 0.4000 | Validloss:2.16256404
```

# test regression model performance
python /~/HDD1/ypang6/workload/DynamicGraph/EvolveGCN-DGL-master/regression.py

# useful command
1. release cuda memories
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9

## TODO
1. Banavar Baseline - speed as node feature - done
2. FAA Baseline - simple regression - done
3. Fine tuning model - done
4. Conformalized prediction set - done
5. Test visualization - done
