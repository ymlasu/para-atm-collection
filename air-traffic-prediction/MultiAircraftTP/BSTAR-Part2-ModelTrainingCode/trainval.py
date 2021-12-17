import argparse
import ast
import os
import torch
import yaml

from src.processor import processor

# Use cudnn deterministic mode and set random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)


def get_parser():
    """User interface for settig the parameters required to start the B-STAR training process, as well as loading the pre-trained B-STAR model to perform testing


    Parameters:
    ----------
    dataset : str
    	The name of the dataset used for training. The default dataset is IFF-ASDEX for KATL 	
    save_dir : str
    	The save directory
    model_dir : str
    	The model saving directory
    config : str
    	Configurations
    using_cuda : bool
    	Use cuda for training with GPU
    test_set : str
    	The selected testset within dataset
    base_dir : str
    	Base directory including the supporting scripts
    save_base_dir : str
    	Directory for saving caches and models
    phase : str
    	Training or Testing phase indicator
    train_model : str
    	The model name to train
    load_model : str
    	Load pre-trained model for testing or training
    model : str
    	The name of the model
    seq_length : int
    	The entire sequence length of time-series
    	Used in constructing the graph structured dataset
    	seq_length is the summation of obs_length and pred_length 
    obs_length : int
    	The input length to the model
    pred_length : int
    	The prediction length of the model
    batch_around_ped : int
    	Number of agents stored as one batch
    	Adjusted based on computer memory limits
    batch_size : int
    	Total number of batches
    test_batch_size : int
    	Size of test batch
    show_step : int
    	Burn-in epoch number for showing training details
    start_test : int
    	Burn-in epoch number for starting test during training
    sample_num : int
    	Number of sample draws during training
    num_epochs : int
    	Total training epochs
    ifshow_detail : bool
    	Show training details
    ifsave_results : bool
    	Save intermediate results
    randomRotate : bool
    	Rotate input data sequence
    neighbor_thred : int
    	Neighboring threshold for dynamical graph determination
    learning_rate : float
    	Learning rate of optimizer
    clip : int
    	Clip value on inputs
    skip : int
    	Time interval in the raw data
    """


    parser = argparse.ArgumentParser(description='BSTAR')
    parser.add_argument('--dataset', default='iffatl')
    parser.add_argument('--save_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--config')
    parser.add_argument('--using_cuda', default=True, type=ast.literal_eval)
    parser.add_argument('--test_set', default='atl0807', type=str,
                        help='Set this value to [atl0801, atl0802, atl0803, atl0804, atl0805, atl0806, atl0807] foe different test set')
    parser.add_argument('--base_dir', default='.', help='Base directory including these scripts.')
    parser.add_argument('--save_base_dir', default='./output/', help='Directory for saving caches and models.')
    parser.add_argument('--phase', default='train', help='Set this value to \'train\' or \'test\'')
    parser.add_argument('--train_model', default='bstar', help='Your model name')
    parser.add_argument('--load_model', default=None, type=str, help="load pretrained model for test or training")
    parser.add_argument('--model', default='bstar.BSTAR')
    parser.add_argument('--seq_length', default=20, type=int)
    parser.add_argument('--obs_length', default=8, type=int)
    parser.add_argument('--pred_length', default=12, type=int)
    parser.add_argument('--batch_around_ped', default=256, type=int)
    parser.add_argument('--batch_size', default=7, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int)
    parser.add_argument('--show_step', default=100, type=int)
    parser.add_argument('--start_test', default=100, type=int)
    parser.add_argument('--sample_num', default=20, type=int)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--ifshow_detail', default=True, type=ast.literal_eval)
    parser.add_argument('--ifsave_results', default=False, type=ast.literal_eval)
    parser.add_argument('--randomRotate', default=True, type=ast.literal_eval,
                        help="=True:random rotation of each trajectory fragment")
    parser.add_argument('--neighbor_thred', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.0015, type=float)
    parser.add_argument('--clip', default=1, type=int)
    parser.add_argument('--skip', default=10, type=int)

    return parser


def load_arg(p):
    # save arg
    if os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s = 1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False


def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()

    p.save_dir = p.save_base_dir + str(p.test_set) + '/'
    p.model_dir = p.save_base_dir + str(p.test_set) + '/' + p.train_model + '/'
    p.config = p.model_dir + '/config_' + p.phase + '.yaml'

    if not load_arg(p):
        save_arg(p)

    args = load_arg(p)

    torch.cuda.set_device(0)

    trainer = processor(args)

    if args.phase == 'test':
        trainer.test()
    else:
        trainer.train()
