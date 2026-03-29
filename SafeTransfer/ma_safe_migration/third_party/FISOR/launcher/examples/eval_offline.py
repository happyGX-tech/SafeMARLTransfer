import os
import sys
sys.path.append('.')
import re
import json
import importlib
import numpy as np
import jax
from absl import app, flags
from ml_collections import ConfigDict
import gymnasium as gym
import wandb
from wandb.errors import CommError

from jaxrl5.agents import FISOR
from jaxrl5.evaluation import evaluate, evaluate_pr
from env.point_robot import PointRobot


FLAGS = flags.FLAGS
flags.DEFINE_string('model_location', '', 'Directory containing config.json and model*.pickle')
flags.DEFINE_string('model_file', '', 'Optional specific model file path; if empty and evaluate_all=False, use latest model')
flags.DEFINE_bool('evaluate_all', True, 'Evaluate all model*.pickle checkpoints in model_location')
flags.DEFINE_integer('eval_episodes', 20, 'Number of evaluation episodes for each checkpoint')
flags.DEFINE_integer('seed', 0, 'Evaluation seed')
flags.DEFINE_bool('render', False, 'Render during evaluation')
flags.DEFINE_string('wandb_project', '', 'Override wandb project')
flags.DEFINE_string('wandb_entity', '', 'Optional wandb entity/team')
flags.DEFINE_string('wandb_mode', 'online', 'wandb mode: online/offline/disabled')


def to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: to_config_dict(v) for k, v in d.items()})
    return d


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


def _resolve_ctce_num_agents(cfg, dataset_path):
    dataset_kwargs = cfg.get('dataset_kwargs', {})
    explicit_num_agents = int(dataset_kwargs.get('num_agents', -1) or -1)
    env_name_num_agents = _infer_num_agents_from_env_name(cfg.get('env_name', ''))
    dataset_num_agents = _infer_num_agents_from_dataset_path(dataset_path)

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

    stale_modules = [m for m in list(sys.modules.keys()) if m == 'safety_gymnasium' or m.startswith('safety_gymnasium.')]
    for module_name in stale_modules:
        del sys.modules[module_name]

    importlib.invalidate_caches()


def _make_env(cfg):
    env_name = cfg['env_name']
    local_dataset_path = cfg.get('dataset_kwargs', {}).get('local_hdf5_path', '')
    use_ctce = _infer_num_agents_from_env_name(env_name) is not None
    if not use_ctce:
        return gym.make(env_name), False

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    _prefer_local_safety_gymnasium(project_root)

    from envs.fsrl_single_agent_wrapper import FSRLCTCESingleAgentEnv

    num_agents = _resolve_ctce_num_agents(cfg, local_dataset_path)
    if num_agents is None:
        raise ValueError('Cannot resolve num_agents for CTCE evaluation.')

    env_id = env_name if _infer_num_agents_from_env_name(env_name) is not None else None
    env = FSRLCTCESingleAgentEnv(
        num_agents=int(num_agents),
        backend='safety_gym',
        env_id=env_id,
        strict_mujoco_consistency=False,
    )
    return env, True


def _select_models(model_location, model_file, evaluate_all):
    if model_file:
        return [model_file]

    files = []
    for name in os.listdir(model_location):
        if name.startswith('model') and name.endswith('.pickle'):
            match = re.search(r'(\d+)', name)
            if match:
                files.append((int(match.group(1)), os.path.join(model_location, name)))

    if not files:
        raise FileNotFoundError(f'No model*.pickle found in {model_location}')

    files.sort(key=lambda x: x[0])
    if evaluate_all:
        return [path for _, path in files]
    return [files[-1][1]]


def _create_agent(cfg, env):
    config_dict = dict(cfg['agent_kwargs'])
    model_cls = config_dict.pop('model_cls')
    config_dict.pop('cost_scale', None)

    if _infer_num_agents_from_env_name(cfg['env_name']) is not None:
        num_agents = _resolve_ctce_num_agents(cfg, cfg.get('dataset_kwargs', {}).get('local_hdf5_path', ''))
        config_dict['decentralized_actor'] = True
        config_dict['num_agents'] = int(num_agents)

    return globals()[model_cls].create(cfg['seed'], env.observation_space, env.action_space, **config_dict)


def _evaluate_one(agent, env, env_name, eval_episodes, render):
    if env_name == 'PointRobot':
        return evaluate_pr(agent, env, eval_episodes)
    return evaluate(agent, env, eval_episodes, render=render)


def _init_wandb_with_fallback(wandb_kwargs):
    try:
        return wandb.init(**wandb_kwargs)
    except CommError as exc:
        message = str(exc)
        # Common failure: invalid entity/team name.
        if 'entity' in message.lower() and 'not found' in message.lower() and 'entity' in wandb_kwargs:
            bad_entity = wandb_kwargs.get('entity', '')
            print(f"[wandb] entity '{bad_entity}' not found. Retrying without entity...")
            retry_kwargs = dict(wandb_kwargs)
            retry_kwargs.pop('entity', None)
            try:
                return wandb.init(**retry_kwargs)
            except CommError as exc2:
                print(f"[wandb] Retry without entity failed: {exc2}")
                print('[wandb] Falling back to offline mode.')
                retry_kwargs['mode'] = 'offline'
                return wandb.init(**retry_kwargs)
        print(f"[wandb] Init failed: {exc}")
        print('[wandb] Falling back to offline mode.')
        retry_kwargs = dict(wandb_kwargs)
        retry_kwargs['mode'] = 'offline'
        retry_kwargs.pop('entity', None)
        return wandb.init(**retry_kwargs)


def _resolve_model_location(flag_value):
    if flag_value:
        return os.path.abspath(flag_value)
    env_value = os.environ.get('MODEL_LOCATION', '')
    if env_value:
        return os.path.abspath(env_value)
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, 'config.json')):
        return os.path.abspath(cwd)
    return ''


def main(_):
    model_location = _resolve_model_location(FLAGS.model_location)
    if model_location == '':
        raise ValueError(
            '--model_location is required. '\
            'You can pass --model_location, or set MODEL_LOCATION, '\
            'or run this script inside a directory containing config.json.'
        )

    config_path = os.path.join(model_location, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'config.json not found in {model_location}')

    with open(config_path, 'r', encoding='utf-8') as file:
        cfg = to_config_dict(json.load(file))

    cfg['seed'] = int(FLAGS.seed)
    eval_project = FLAGS.wandb_project or cfg.get('project', 'FISOR-eval') or 'FISOR-eval'
    run_name = f"eval_{os.path.basename(model_location)}"

    wandb_kwargs = {
        'project': eval_project,
        'name': run_name,
        'group': cfg.get('group', cfg.get('env_name', 'eval')),
        'config': {
            'model_location': model_location,
            'eval_episodes': int(FLAGS.eval_episodes),
            'evaluate_all': bool(FLAGS.evaluate_all),
            'env_name': cfg.get('env_name', ''),
        },
        'mode': FLAGS.wandb_mode,
        'dir': model_location,
    }
    if FLAGS.wandb_entity:
        wandb_kwargs['entity'] = FLAGS.wandb_entity

    _init_wandb_with_fallback(wandb_kwargs)

    env, _ = _make_env(cfg)
    base_agent = _create_agent(cfg, env)
    model_paths = _select_models(model_location, FLAGS.model_file, FLAGS.evaluate_all)

    results = []
    for idx, model_path in enumerate(model_paths):
        eval_agent = base_agent.load(model_path)
        metrics = _evaluate_one(
            eval_agent,
            env,
            cfg['env_name'],
            int(FLAGS.eval_episodes),
            bool(FLAGS.render),
        )
        metrics = {k: float(v) for k, v in metrics.items()}
        metrics['model_path'] = model_path
        metrics['model_name'] = os.path.basename(model_path)
        metrics['model_index'] = idx
        results.append(metrics)

        log_payload = {'global_step': idx}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_payload[f'eval/{key}'] = float(value)
        wandb.log(log_payload, step=idx)
        print(f"[eval] {metrics['model_name']}: return={metrics.get('return', 0.0):.4f}, cost={metrics.get('cost', 0.0):.4f}, len={metrics.get('episode_len', 0.0):.2f}")

    summary = {
        'best_return': max(results, key=lambda x: x.get('return', -1e30)),
        'min_cost': min(results, key=lambda x: x.get('cost', 1e30)),
        'num_models': len(results),
    }

    summary_path = os.path.join(model_location, 'eval_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump({'results': results, 'summary': summary}, file, indent=2, ensure_ascii=False)

    wandb.summary['best_return'] = float(summary['best_return'].get('return', 0.0))
    wandb.summary['best_return_model'] = summary['best_return']['model_name']
    wandb.summary['min_cost'] = float(summary['min_cost'].get('cost', 0.0))
    wandb.summary['min_cost_model'] = summary['min_cost']['model_name']
    wandb.summary['num_models'] = int(summary['num_models'])
    wandb.finish()

    env.close()
    print(f"Saved evaluation summary to: {summary_path}")


if __name__ == '__main__':
    app.run(main)
