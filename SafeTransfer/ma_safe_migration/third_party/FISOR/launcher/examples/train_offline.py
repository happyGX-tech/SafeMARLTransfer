import os
import sys
sys.path.append('.')
import random
import re
import importlib
import warnings
import numpy as np

warnings.filterwarnings(
    'ignore',
    message=r'.*Overriding environment .* already in registry.*',
    category=UserWarning,
)

# Avoid JAX pre-allocating most memory on startup.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
from absl import app, flags
import datetime
import yaml
from ml_collections import config_flags, ConfigDict
import wandb
from tqdm.auto import trange  # noqa
import gymnasium as gym
from env.env_list import env_list
from env.point_robot import PointRobot
from jaxrl5.wrappers import wrap_gym
from jaxrl5.agents import FISOR
from jaxrl5.data.dsrl_datasets import DSRLDataset
from jaxrl5.evaluation import evaluate, evaluate_pr
import json


FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 30, 'Choose env')
flags.DEFINE_float('ratio', 1.0, 'dataset ratio')
flags.DEFINE_string('project', '', 'project name for wandb')
flags.DEFINE_string('experiment_name', '', 'experiment name for wandb')
flags.DEFINE_string('dataset_path', '', 'Optional local HDF5 dataset path to bypass URL loading')
flags.DEFINE_string('custom_env_name', '', 'Optional env name override, e.g. sg_ant_goal_n4 or SafetyAntMultiGoalN4-v0')
flags.DEFINE_integer('num_agents', -1, 'Optional explicit num_agents for variable-agent CTCE datasets')
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config


def _infer_num_agents_from_env_name(env_name):
    if not env_name:
        return None
    for pattern in (r'sg_ant_goal_n(\d+)', r'SafetyAntMultiGoalN(\d+)-v0'):
        match = re.fullmatch(pattern, env_name)
        if match:
            return int(match.group(1))
    return None


def _infer_num_agents_from_dataset_path(dataset_path):
    if not dataset_path:
        return None
    basename = os.path.basename(dataset_path)
    match = re.search(r'ctce_n(\d+)', basename)
    if match:
        return int(match.group(1))
    return None


def _resolve_ctce_num_agents(details, local_dataset_path):
    explicit_num_agents = details['dataset_kwargs'].get('num_agents', -1)
    explicit_num_agents = int(explicit_num_agents) if explicit_num_agents is not None else -1
    env_name_num_agents = _infer_num_agents_from_env_name(details.get('env_name', ''))
    dataset_num_agents = _infer_num_agents_from_dataset_path(local_dataset_path)

    if explicit_num_agents > 0 and env_name_num_agents is not None and explicit_num_agents != env_name_num_agents:
        raise ValueError(
            f"num_agents mismatch: dataset_kwargs.num_agents={explicit_num_agents}, env_name implies {env_name_num_agents}."
        )
    if explicit_num_agents > 0 and dataset_num_agents is not None and explicit_num_agents != dataset_num_agents:
        raise ValueError(
            f"num_agents mismatch: dataset_kwargs.num_agents={explicit_num_agents}, dataset path implies ctce_n{dataset_num_agents}."
        )
    if env_name_num_agents is not None and dataset_num_agents is not None and env_name_num_agents != dataset_num_agents:
        raise ValueError(
            f"num_agents mismatch: env_name implies {env_name_num_agents}, dataset path implies ctce_n{dataset_num_agents}."
        )

    if explicit_num_agents > 0:
        return explicit_num_agents
    if env_name_num_agents is not None:
        return env_name_num_agents
    if dataset_num_agents is not None:
        return dataset_num_agents
    return None


def _prefer_local_safety_gymnasium(project_root):
    workspace_root = os.path.abspath(os.path.join(project_root, '..', '..'))
    candidates = [
        os.environ.get('SAFETY_GYM_LOCAL_PATH', ''),
        os.path.join(workspace_root, 'SRL', 'safety-gymnasium-main'),
        os.path.join(project_root, 'third_party', 'safety-gymnasium-main'),
        os.path.join(project_root, 'external', 'safety-gymnasium-main'),
    ]
    local_path = next((p for p in candidates if p and os.path.isdir(p)), None)
    if local_path is None:
        return

    os.environ['SAFETY_GYM_LOCAL_PATH'] = local_path
    if local_path in sys.path:
        sys.path.remove(local_path)
    sys.path.insert(0, local_path)

    # If safety_gymnasium was imported from site-packages earlier, reload from local source.
    stale_modules = [m for m in list(sys.modules.keys()) if m == 'safety_gymnasium' or m.startswith('safety_gymnasium.')]
    for module_name in stale_modules:
        del sys.modules[module_name]

    importlib.invalidate_caches()


def _make_env(details, local_dataset_path):
    env_name = details['env_name']
    use_ctce = bool(local_dataset_path) and (
        _infer_num_agents_from_env_name(env_name) is not None
        or _infer_num_agents_from_dataset_path(local_dataset_path) is not None
        or int(details['dataset_kwargs'].get('num_agents', -1)) > 0
    )
    if not use_ctce:
        return gym.make(env_name), False

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    _prefer_local_safety_gymnasium(project_root)

    from envs.fsrl_single_agent_wrapper import FSRLCTCESingleAgentEnv

    num_agents = _resolve_ctce_num_agents(details, local_dataset_path)
    if num_agents is None:
        raise ValueError(
            "Cannot resolve num_agents for CTCE training. Please provide --num_agents, "
            "or use env name sg_ant_goal_nN / SafetyAntMultiGoalNN-v0, "
            "or include ctce_nN in dataset file name."
        )

    env_id = env_name if _infer_num_agents_from_env_name(env_name) is not None else None
    env = FSRLCTCESingleAgentEnv(
        num_agents=num_agents,
        backend='safety_gym',
        env_id=env_id,
        strict_mujoco_consistency=False,
    )
    return env, True


def call_main(details):
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']
    wandb.init(project=details['project'], name=details['experiment_name'], group=details['group'], config=details['agent_kwargs'])
    local_dataset_path = details['dataset_kwargs'].get('local_hdf5_path', '')

    if details['env_name'] == 'PointRobot':
        point_robot_data = local_dataset_path or details['dataset_kwargs']['pr_data']
        assert point_robot_data is not None and point_robot_data != '', "No data for Point Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], data_location=point_robot_data, cost_scale=details['dataset_kwargs']['cost_scale'])
        is_ctce_env = False
    else:
        env, is_ctce_env = _make_env(details, local_dataset_path)
        if is_ctce_env and details['batch_size'] > 256:
            print(f"[CTCE] Reduce batch_size from {details['batch_size']} to 256 to avoid JAX OOM.")
            details['batch_size'] = 256
        ds = DSRLDataset(
            env,
            critic_type=details['agent_kwargs']['critic_type'],
            data_location=local_dataset_path if local_dataset_path else None,
            cost_scale=details['dataset_kwargs']['cost_scale'],
            ratio=details['ratio'],
        )
        env_max_steps = getattr(env, '_max_episode_steps', None)
        if env_max_steps is None:
            env_max_steps = getattr(getattr(env, 'ma_env', None), 'max_episode_steps', None)
        if env_max_steps is None:
            spec = getattr(env, 'spec', None)
            env_max_steps = getattr(spec, 'max_episode_steps', None)
        if env_max_steps is None:
            env_max_steps = 1000
        if not is_ctce_env:
            env = wrap_gym(env, cost_limit=details['agent_kwargs']['cost_limit'])
        if not is_ctce_env and hasattr(env, 'max_episode_reward') and hasattr(env, 'min_episode_reward'):
            ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    ds.seed(details["seed"])

    config_dict = dict(details['agent_kwargs'])
    config_dict['env_max_steps'] = env_max_steps
    if is_ctce_env:
        resolved_num_agents = getattr(env, 'num_agents', None)
        if resolved_num_agents is None:
            resolved_num_agents = _resolve_ctce_num_agents(details, local_dataset_path)
        config_dict['decentralized_actor'] = True
        config_dict['num_agents'] = int(resolved_num_agents)

    model_cls = config_dict.pop("model_cls") 
    config_dict.pop("cost_scale") 
    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )


    save_time = 1
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        if is_ctce_env:
            sample = ds.sample(details['batch_size'])
        else:
            sample = ds.sample_jax(details['batch_size'])
        agent, info = agent.update(sample)
        
        if i % details['log_interval'] == 0:
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)

        # if i % details['eval_interval'] == 0 and i > 0:
        if (not is_ctce_env) and i % details['eval_interval'] == 0:
            agent.save(f"./results/{details['group']}/{details['experiment_name']}", save_time)
            save_time += 1
            if details['env_name'] == 'PointRobot':
                eval_info = evaluate_pr(agent, env, details['eval_episodes'])
            else:
                eval_info = evaluate(agent, env, details['eval_episodes'])
            if details['env_name'] != 'PointRobot' and hasattr(env, 'get_normalized_score'):
                eval_info["normalized_return"], eval_info["normalized_cost"] = env.get_normalized_score(eval_info["return"], eval_info["cost"])
            wandb.log({f"eval/{k}": v for k, v in eval_info.items()}, step=i)


def main(_):
    parameters = FLAGS.config
    if FLAGS.project != '':
        parameters['project'] = FLAGS.project
    parameters['env_name'] = FLAGS.custom_env_name if FLAGS.custom_env_name != '' else env_list[FLAGS.env_id]
    parameters['ratio'] = FLAGS.ratio
    if FLAGS.num_agents > 0:
        parameters['dataset_kwargs']['num_agents'] = FLAGS.num_agents
    if FLAGS.dataset_path != '':
        parameters['dataset_kwargs']['local_hdf5_path'] = FLAGS.dataset_path
    parameters['group'] = parameters['env_name']

    parameters['experiment_name'] = parameters['agent_kwargs']['sampling_method'] + '_' \
                                + parameters['agent_kwargs']['actor_objective'] + '_' \
                                + parameters['agent_kwargs']['critic_type'] + '_N' \
                                + str(parameters['agent_kwargs']['N']) + '_' \
                                + parameters['agent_kwargs']['extract_method'] if FLAGS.experiment_name == '' else FLAGS.experiment_name
    parameters['experiment_name'] += '_' + str(datetime.date.today()) + '_s' + str(parameters['seed']) + '_' + str(random.randint(0,1000))

    if parameters['env_name'] == 'PointRobot':
        parameters['max_steps'] = 100001
        parameters['batch_size'] = 1024
        parameters['eval_interval'] = 25000
        parameters['agent_kwargs']['cost_temperature'] = 2
        parameters['agent_kwargs']['reward_temperature'] = 5
        parameters['agent_kwargs']['cost_ub'] = 150
        parameters['agent_kwargs']['N'] = 8

    print(parameters)

    if not os.path.exists(f"./results/{parameters['group']}/{parameters['experiment_name']}"):
        os.makedirs(f"./results/{parameters['group']}/{parameters['experiment_name']}")
    with open(f"./results/{parameters['group']}/{parameters['experiment_name']}/config.json", "w") as f:
        json.dump(to_dict(parameters), f, indent=4)
    
    call_main(parameters)


if __name__ == '__main__':
    app.run(main)
