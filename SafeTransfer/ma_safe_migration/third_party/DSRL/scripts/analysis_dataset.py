import argparse
import glob
import os
import os.path as osp

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

try:
    from .utils import get_trajectory_info
except ImportError:
    current_dir = osp.dirname(osp.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from utils import get_trajectory_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--output', '-o', type=str, default='cr-plot.png')
    parser.add_argument('--maxlen', type=int, default=50000000)
    parser.add_argument('--pattern', type=str, default='*.hdf5')
    args = parser.parse_args()

    root_dir = osp.abspath(args.root)
    if not osp.isdir(root_dir):
        raise NotADirectoryError(f"Root directory not found: {root_dir}")

    file_paths = glob.glob(os.path.join(root_dir, '**', args.pattern), recursive=True)
    file_paths = sorted(file_paths)
    print(f"search root: {root_dir}")
    print(f"matched files: {len(file_paths)}")
    if not file_paths:
        raise FileNotFoundError(
            f"No files matched pattern '{args.pattern}' under {root_dir}. "
            "Use --pattern to adjust file matching."
        )

    for file_path in file_paths:
        dir_path = os.path.dirname(file_path)
        print("reading from ... ", dir_path)
        with h5py.File(file_path, 'r') as data:

            keys = [
                'observations', 'next_observations', 'actions', 'rewards', 'costs',
                'terminals', 'timeouts'
            ]

            dataset_dict = {}
            for k in keys:
                
                combined = np.array(data[k])[:args.maxlen]
                print(k, combined.shape)
                dataset_dict[k] = combined
            #单独统计terminals和timeouts中为True的数量
            terminals_true = np.sum(dataset_dict['terminals'])
            timeouts_true = np.sum(dataset_dict['timeouts'])
            print(f"Number of terminals: {terminals_true}")
            print(f"Number of timeouts: {timeouts_true}")
            rew_ret, cost_ret, start_index, end_index = get_trajectory_info(dataset_dict)

            print(f"Total number of trajectories: {len(rew_ret)}")

            plt.scatter(cost_ret, rew_ret, alpha=0.5,color='blue', edgecolors='w', s=50)
            plt.xlabel("Costs Returns")
            plt.ylabel("Rewards Returns")
            output_path = os.path.join(dir_path, args.output)
            plt.savefig(output_path)
            print(f"saved: {output_path}")
            plt.clf()
            print(len(cost_ret))
            #打印cost_ret中大于0的数量
            cost_ret_positive = [cost for cost in cost_ret if cost > 0]
            print(f"Number of positive costs: {len(cost_ret_positive)}")
