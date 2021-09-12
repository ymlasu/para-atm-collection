# Data-Driven Uncertainty Aware Multi-Aircraft Trajectory Prediction in the Near Terminal Airspace 


# B-STAR
Code repository for the AAAI submission of paper entitled: Bayesian Spatio-Temporal Graph Transformer Network for Near-Terminal Multi-Aircraft Trajectory Prediction with Uncertainties


### Environment

```bash
pip install numpy==1.18.1
pip install torch==1.7.0
pip install pyyaml=5.3.1
pip install tqdm=4.45.0
```

### Data
The data source used are ASDE-X data from Sherlock. However, we anonymized the data by,

- Remove the real-world unix timestamp and replace them by absolute time steps (e.g. 5, 10, 15, 20).
- The flight callsign are masked with a unique agent id (integer).
- For this experiment, only four columns of ASDE-X needed. They are time, id, latitude, longitude.

The code used for processing and anonymized the raw ASDE-X data can be found in Part 1 Data Processing Demo. Data are saved in ```/data/iff/atl/2019080x/true_pos_.csv```


### To Train an Example
This command is to train model using the ASDE-X data from Aug 1st, 2019 to Aug 6th, 2019, and test the trained model with data from Aug 7th, 2019.

```
python trainval.py --num_epochs 300 --start_test 250 --neighbor_thred 10 --skip 5 --seq_length 20 --obs_length 12 --pred_length 8 --randomRotate False --learning_rate 0.0015 --sample_num 20
```

The model will be trained for 300 epochs, and the testing start at epoch 250, with a learning rate of 0.0015. In the test phase, the trained model will be sampled 20 times.

And with the following paramaters, 
- The neighboring aircraft threshold is 10km(~5nm). 
- The timestamp in the processed flight data is 5 seconds. 
- The total length of the sequence is 20 timestamps, where observation is 12 timestamps, prediction is 8 timestamps.

During training, the trained model wth a new best FDE on the test dataset will be saved in the output folder.


### Source Code
In ```\src```, there are multiple Python scripts,

- ```utils.py```: Data pre-processing before training. For instance, using the previous command, there will be 24 training batches and 7 testing batches.
- ```lrt_linear.py```: Bayesian Linear Layer used to build the decoder using locao reparameterization trick.
- ```multi_attention_forward.py```: The multi-head attention layer.
- ```bstar.py```: The code to build the B-STAR architecture.
- ```processor.py```: Training and testing function.


### Reference

Code of Part 2 borrowed heavily from [here](https://github.com/Majiker/STAR)
