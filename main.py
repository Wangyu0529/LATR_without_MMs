import argparse
from utils.utils import *
from experiments.ddp import *
from experiments.runner import *

def get_args(json_file_path):
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    args = argparse.Namespace()
    for key, value in json_data.items():
        if key == "anchor_y_steps" or key == "anchor_y_steps_dense":
            # Convert list to numpy array
            value = np.linspace(value[0], value[1], value[2])
        elif isinstance(value, list):
            # Convert list to numpy array
            value = np.array(value)
        setattr(args, key, value)
    return args

def main():
    json_file_path = '/root/WY/LATR_without_mms/config/experience.json'
    args = get_args(json_file_path)
    
    ddp_init(args)
    runner = Runner(args)
    if args.is_train:
        print('Training...')
        runner.train()
    else:
        print('Evaluating...')
        runner.eval()

if __name__ == '__main__':
    main()
