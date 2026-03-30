import argparse
from calendar import c
from collections import defaultdict
import os
import os.path as osp
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

try:
    from .utils import get_trajectory_info, filter_trajectory
except ImportError:
    current_dir = osp.dirname(osp.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from utils import get_trajectory_info, filter_trajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*')
    parser.add_argument('--output', '-o', type=str, default='car-circle')
    parser.add_argument('--dir', '-d', type=str, default='log')
    parser.add_argument('--cmax', type=int, default=100)
    parser.add_argument('--cmin', type=int, default=0)
    parser.add_argument('--rmax', type=float, default=1000)
    parser.add_argument('--rmin', type=float, default=0)
    parser.add_argument('--cbins', type=int, default=60)
    parser.add_argument('--rbins', type=int, default=50)
    parser.add_argument('--npb', type=int, default=1000)
    parser.add_argument('--no_filter', action="store_true", help='Disable grid filtering and keep all trajectories.')
    parser.add_argument('--preserve_cost_le', type=float, default=None, help='Always keep trajectories with episode cost <= this threshold before filtering others.')
    parser.add_argument('--save', action="store_true")
    args = parser.parse_args()

    hfiles = []
    for file in args.files:
        hfiles.append(h5py.File(file, 'r'))

    keys = [
        'observations', 'next_observations', 'actions', 'rewards', 'costs', 'terminals',
        'timeouts'
    ]
    float_32_keys = ['observations', 'next_observations', 'actions', 'rewards', 'costs']

    print("*" * 10, "concatenating dataset from", "*" * 10)
    for file in args.files:
        print("*" * 10, file, "*" * 10)
    dataset_dict = {}
    for k in keys:
        d = [hfile[k] for hfile in hfiles]
        combined = np.concatenate(d, axis=0)
        # convert the data to the same float 32 data format
        if k in float_32_keys:
            combined = combined.astype(np.float32)
        print(k, combined.shape, combined.dtype)
        dataset_dict[k] = combined
    print("*" * 10, "dataset concatenation finished", "*" * 10)

    # process the dataset to the SDT format:
    # traj[i] is the i-th trajectory (a dict)
    rew_ret, cost_ret, start_index, end_index = get_trajectory_info(dataset_dict)
    traj = []
    for i in trange(len(rew_ret), desc="Processing trajectories..."):
        start = start_index[i]
        end = end_index[i] + 1
        one_traj = {k: dataset_dict[k][start:end] for k in keys}
        traj.append(one_traj)
    print(f"Total number of trajectories: {len(traj)}")

    # plot the original dataset cost-reward figure
    output_name = args.output + '-' + str(int(args.cmax))
    plt.scatter(cost_ret, rew_ret)
    plt.xlabel("Costs")
    plt.ylabel("Rewards")
    output_path = osp.join("fig", output_name + "_before_filter.png")
    if not osp.exists("fig"):
        os.makedirs("fig")
    plt.savefig(output_path)
    plt.clf()

    original_traj_num = len(traj)

    # downsampling the trajectories by grid filter
    if args.no_filter:
        print('[INFO] --no_filter enabled: keep all trajectories without grid filtering.')
    else:
        if args.preserve_cost_le is not None:
            keep_cost = float(args.preserve_cost_le)
            keep_indices = [i for i, c in enumerate(cost_ret) if c <= keep_cost]
            filter_indices = [i for i, c in enumerate(cost_ret) if c > keep_cost]

            preserved_cost = [cost_ret[i] for i in keep_indices]
            preserved_rew = [rew_ret[i] for i in keep_indices]
            preserved_traj = [traj[i] for i in keep_indices]

            filter_cost = [cost_ret[i] for i in filter_indices]
            filter_rew = [rew_ret[i] for i in filter_indices]
            filter_traj = [traj[i] for i in filter_indices]

            print(
                f"[INFO] preserve_cost_le={keep_cost}: keep {len(preserved_traj)} low-cost trajectories, "
                f"apply grid filter on {len(filter_traj)} trajectories."
            )

            if len(filter_traj) > 0:
                filter_cost, filter_rew, filter_traj = filter_trajectory(
                    filter_cost,
                    filter_rew,
                    filter_traj,
                    cost_min=args.cmin,
                    cost_max=args.cmax,
                    rew_min=args.rmin,
                    rew_max=args.rmax,
                    cost_bins=args.cbins,
                    rew_bins=args.rbins,
                    max_num_per_bin=args.npb,
                )

            cost_ret = preserved_cost + filter_cost
            rew_ret = preserved_rew + filter_rew
            traj = preserved_traj + filter_traj
        else:
            cost_ret, rew_ret, traj = filter_trajectory(
                cost_ret,
                rew_ret,
                traj,
                cost_min=args.cmin,
                cost_max=args.cmax,
                rew_min=args.rmin,
                rew_max=args.rmax,
                cost_bins=args.cbins,
                rew_bins=args.rbins,
                max_num_per_bin=args.npb,
            )

    print(f"Num of trajectories after filtering: {len(traj)}")
    print(f"Trajectory keep ratio: {len(traj)}/{original_traj_num} = {len(traj) / max(1, original_traj_num):.4f}")

    # plot the filtered dataset cost-reward figure
    plt.scatter(cost_ret, rew_ret)
    plt.xlabel("Costs")
    plt.ylabel("Rewards")
    output_path = osp.join("fig", output_name + "_after_filter.png")
    plt.savefig(output_path)

    # process the trajectory data back to the d4rl data format:
    dataset = defaultdict(list)
    for d in traj:
        for k in keys:
            dataset[k].append(d[k])
    for k in keys:
        dataset[k] = np.squeeze(np.concatenate(dataset[k], axis=0))
        print(k, np.array(dataset[k]).shape, dataset[k].dtype)

    output_name = args.output + '-' + str(int(args.cmax)) + '-' + str(len(traj))

    # store the data
    if args.save:
        if not osp.exists(args.dir):
            os.makedirs(args.dir)
        output_name = output_name + ".hdf5"
        output_file = osp.join(args.dir, output_name)
        outf = h5py.File(output_file, 'w')
        for k in keys:
            outf.create_dataset(k, data=dataset[k], compression='gzip')
        outf.close()
