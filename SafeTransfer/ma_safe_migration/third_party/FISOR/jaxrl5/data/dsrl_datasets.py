import os
import gymnasium as gym
import dsrl
import numpy as np
from jaxrl5.data.dataset import Dataset
import h5py


class DSRLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5, critic_type="qc", data_location=None, cost_scale=1., ratio = 1.0):
        def _load_local_hdf5(h5_path, local_ratio=1.0):
            local_dataset = {}
            with h5py.File(h5_path, 'r') as f:
                keys = set(f.keys())

                def _read_array(key):
                    arr = f[key]
                    if local_ratio is not None and 0 < float(local_ratio) < 1.0 and arr.ndim > 0:
                        limit = max(1, int(arr.shape[0] * float(local_ratio)))
                        return np.array(arr[:limit])
                    return np.array(arr)

                # PointRobot legacy format.
                if {'state', 'action', 'next_state', 'reward'}.issubset(keys):
                    local_dataset["observations"] = _read_array('state')
                    local_dataset["actions"] = _read_array('action')
                    local_dataset["next_observations"] = _read_array('next_state')
                    local_dataset["rewards"] = _read_array('reward')
                    if 'done' in keys:
                        local_dataset["dones"] = _read_array('done')
                    elif {'terminals', 'timeouts'}.issubset(keys):
                        local_dataset["dones"] = np.logical_or(
                            _read_array('terminals'),
                            _read_array('timeouts'),
                        ).astype(np.float32)
                    else:
                        raise KeyError("Local dataset missing done/terminals-timeouts fields.")

                    if 'h' in keys:
                        local_dataset['costs'] = _read_array('h')
                    elif 'costs' in keys:
                        local_dataset['costs'] = _read_array('costs')
                    elif 'cost' in keys:
                        local_dataset['costs'] = _read_array('cost')
                    else:
                        raise KeyError("Local dataset missing h/costs/cost field.")
                    return local_dataset

                # DSRL/CTCE standard format.
                required_keys = {'observations', 'actions', 'next_observations', 'rewards'}
                if not required_keys.issubset(keys):
                    raise KeyError(
                        "Local dataset must contain observations/actions/next_observations/rewards "
                        "or PointRobot state/action/next_state/reward format."
                    )

                local_dataset["observations"] = _read_array('observations')
                local_dataset["actions"] = _read_array('actions')
                local_dataset["next_observations"] = _read_array('next_observations')
                local_dataset["rewards"] = _read_array('rewards')

                if 'dones' in keys:
                    local_dataset["dones"] = _read_array('dones')
                elif {'terminals', 'timeouts'}.issubset(keys):
                    local_dataset["dones"] = np.logical_or(
                        _read_array('terminals'),
                        _read_array('timeouts'),
                    ).astype(np.float32)
                else:
                    raise KeyError("Local dataset missing dones or terminals+timeouts fields.")

                if 'costs' in keys:
                    local_dataset['costs'] = _read_array('costs')
                elif 'h' in keys:
                    local_dataset['costs'] = _read_array('h')
                elif 'cost' in keys:
                    local_dataset['costs'] = _read_array('cost')
                else:
                    raise KeyError("Local dataset missing costs/h/cost field.")

            return local_dataset

        if data_location is not None:
            dataset_path = os.path.abspath(os.path.expanduser(data_location))
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Local dataset not found: {dataset_path}")

            print('=========Data loading=========')
            print('Load local data from:', dataset_path)
            try:
                dataset_dict = _load_local_hdf5(dataset_path, local_ratio=ratio)
            except np.core._exceptions._ArrayMemoryError as exc:
                raise MemoryError(
                    "Failed to load local dataset into memory. "
                    "Try a smaller --ratio (e.g. 0.1) for local HDF5 training. "
                    f"Original error: {exc}"
                ) from exc

            env_max_episode_steps = getattr(env, '_max_episode_steps', None)
            if env_max_episode_steps is None:
                env_max_episode_steps = getattr(getattr(env, 'ma_env', None), 'max_episode_steps', None)
            if env_max_episode_steps is None:
                spec = getattr(env, 'spec', None)
                env_max_episode_steps = getattr(spec, 'max_episode_steps', None)
            if env_max_episode_steps is None:
                env_max_episode_steps = 1000

            print('env_max_episode_steps', env_max_episode_steps)
            print('mean_episode_reward', env_max_episode_steps * np.mean(dataset_dict['rewards']))
            print('mean_episode_cost', env_max_episode_steps * np.mean(dataset_dict['costs']))

        else:
            # DSRL
            if ratio == 1.0:
                dataset_dict = env.get_dataset()
            else:
                _, dataset_name = os.path.split(env.dataset_url)
                file_list = dataset_name.split('-')
                ratio_num = int(float(file_list[-1].split('.')[0]) * ratio)
                dataset_ratio = '-'.join(file_list[:-1]) + '-' + str(ratio_num) + '-' + str(ratio) + '.hdf5'
                dataset_ratio_path = os.path.join('data', dataset_ratio)
                if not os.path.exists(dataset_ratio_path):
                    raise FileNotFoundError(
                        f"Ratio dataset not found: {dataset_ratio_path}. "
                        "Please place the processed file under ./data or set --dataset_path."
                    )
                dataset_dict = env.get_dataset(dataset_ratio_path)
            print('max_episode_reward', env.max_episode_reward, 
                'min_episode_reward', env.min_episode_reward,
                'mean_episode_reward', env._max_episode_steps * np.mean(dataset_dict['rewards']))
            print('max_episode_cost', env.max_episode_cost, 
                'min_episode_cost', env.min_episode_cost,
                'mean_episode_cost', env._max_episode_steps * np.mean(dataset_dict['costs']))
            print('data_num', dataset_dict['actions'].shape[0])
            dataset_dict['dones'] = np.logical_or(dataset_dict["terminals"],
                                                dataset_dict["timeouts"]).astype(np.float32)
            del dataset_dict["terminals"]
            del dataset_dict['timeouts']

        if critic_type == "hj":
            dataset_dict['costs'] = np.where(dataset_dict['costs'] > 0, 1 * cost_scale, -1)

        # Normalize scalar fields to (N, ) and validate env-data shape alignment.
        for key in ('rewards', 'costs', 'dones'):
            if key in dataset_dict and dataset_dict[key].ndim == 2 and dataset_dict[key].shape[1] == 1:
                dataset_dict[key] = dataset_dict[key][:, 0]

        obs_shape = getattr(env.observation_space, 'shape', None)
        if obs_shape is not None and len(obs_shape) > 0:
            expected_obs_shape = tuple(obs_shape)
            actual_obs_shape = tuple(dataset_dict['observations'].shape[1:])
            if actual_obs_shape != expected_obs_shape:
                raise ValueError(
                    "Observation shape mismatch between env and dataset. "
                    f"env={expected_obs_shape}, dataset={actual_obs_shape}. "
                    "For variable-agent CTCE training, ensure env_name/num_agents matches the dataset."
                )

        act_shape = getattr(env.action_space, 'shape', None)
        if act_shape is not None and len(act_shape) > 0:
            expected_act_shape = tuple(act_shape)
            actual_act_shape = tuple(dataset_dict['actions'].shape[1:])
            if actual_act_shape != expected_act_shape:
                raise ValueError(
                    "Action shape mismatch between env and dataset. "
                    f"env={expected_act_shape}, dataset={actual_act_shape}. "
                    "For variable-agent CTCE training, ensure env_name/num_agents matches the dataset."
                )

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["masks"] = 1.0 - dataset_dict['dones']
        del dataset_dict['dones']

        super().__init__(dataset_dict)
