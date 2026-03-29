import os
import sys
sys.path.append('.')
from absl import app, flags
import re
import json
import importlib
import numpy as np
import gymnasium as gym
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from matplotlib import colors
import jax
from env.point_robot import PointRobot
from jaxrl5.agents import FISOR


FLAGS = flags.FLAGS
flags.DEFINE_string('model_location', '', 'Model directory containing config.json and model*.pickle')
flags.DEFINE_string('model_file', '', 'Optional explicit model file path. If empty, use latest checkpoint')
flags.DEFINE_integer('seed', 0, 'Seed used for environment reset in visualization')
flags.DEFINE_integer('x_index', 0, 'Observation index used as x-axis in CTCE value slice')
flags.DEFINE_integer('y_index', 1, 'Observation index used as y-axis in CTCE value slice')
flags.DEFINE_float('span', 3.0, 'Half width of visualization range [-span, span] for CTCE value slice')
flags.DEFINE_integer('grid_size', 121, 'Grid size for CTCE value slice visualization')


def to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: to_config_dict(v) for k, v in d.items()})
    return d


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


def _resolve_ctce_num_agents(cfg):
    dataset_path = cfg.get('dataset_kwargs', {}).get('local_hdf5_path', '')
    explicit_num_agents = int(cfg.get('dataset_kwargs', {}).get('num_agents', -1) or -1)
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
    env_name = cfg.get('env_name', 'PointRobot')
    if env_name == 'PointRobot':
        return PointRobot(id=0, seed=int(FLAGS.seed)), False

    if _infer_num_agents_from_env_name(env_name) is None:
        return gym.make(env_name), False

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    _prefer_local_safety_gymnasium(project_root)

    from envs.fsrl_single_agent_wrapper import FSRLCTCESingleAgentEnv

    num_agents = _resolve_ctce_num_agents(cfg)
    if num_agents is None:
        raise ValueError('Cannot resolve num_agents for CTCE visualization.')

    env = FSRLCTCESingleAgentEnv(
        num_agents=int(num_agents),
        backend='safety_gym',
        env_id=env_name,
        strict_mujoco_consistency=False,
    )
    return env, True

hazard_position_list = [np.array([0.4, -1.2]), np.array([-0.4, 1.2])]

label_size = 18
legend_size = 30
ticks_size = 18
location = -0.3
width = 0.5

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': label_size}

def plot_pr_pic(ax, agent, v, theta, cb=False):
    ## generate batch obses
    x1 = np.linspace(-3.0, 3.0, 201)
    x2 = np.linspace(-3.0, 3.0, 201)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    flatten_x1 = x1_grid.ravel()
    flatten_x2 = x2_grid.ravel()
    batch_obses = np.zeros((len(flatten_x1), 11), dtype=np.float32)  # (201*201, 11)
    assert batch_obses.shape == (201*201, 11)
    batch_obses[:, 0] = flatten_x1
    batch_obses[:, 1] = flatten_x2

    batch_obses[:, 2] = v * np.ones_like(flatten_x1)
    thetas = theta * np.ones_like(flatten_x1)
    batch_obses[:, 3] = np.cos(thetas)
    batch_obses[:, 4] = np.sin(thetas)

    c = np.cos(theta)
    s = np.sin(theta)

    rot_mat = np.array([[c, -s],
                        [s, c]], dtype=np.float32)

    k = 0

    for hazard_pos in hazard_position_list:

        pos = (hazard_pos[:2] - batch_obses[:,:2]) @ rot_mat  # (B, 2)
        x = pos[:,0]
        y = pos[:,1]
        hazard_vec = x + 1j * y

        dist = np.abs(hazard_vec)
        angle = np.angle(hazard_vec)

        batch_obses[:,5+k*3] = dist
        batch_obses[:,6+k*3] = np.cos(angle)
        batch_obses[:,7+k*3] = np.sin(angle)

        k += 1
    
    '''
    safe value
    '''
    safe_value = agent.safe_value.apply_fn({"params": agent.safe_value.params}, jax.device_put(batch_obses))
    value = safe_value

    value_flatten = np.asarray(value)
    value_square = value_flatten.reshape(x1_grid.shape)
        
    '''
    draw hj
    '''
    norm = colors.Normalize(vmin=-3.5, vmax=1.01)
    
    ct = ax.contourf(
        x1_grid, x2_grid, value_square,
        norm=norm,
        levels=30,
        cmap='rainbow',
    )

    ct_line = ax.contour(
        x1_grid, x2_grid, value_square,
        levels=[0], colors='#32ABD6',
        linewidths=2.0, linestyles='solid'
    )
    ax.clabel(ct_line, inline=True, fontsize=15, fmt=r'0',)

    if cb==True:
        cb = plt.colorbar(ct, ax=ax, shrink=0.8, pad=0.02, ticks=np.linspace(-3.2, 0.8, 6))
        cb.ax.tick_params(labelsize=ticks_size)

        cbarlabels = cb.ax.get_yticklabels() 
        [label.set_fontname('Times New Roman') for label in cbarlabels]

    arrow_x1 = np.linspace(-1.8, 1.8, 3)
    arrow_x2 = np.linspace(-1.8, 1.8, 3)
    ax1_grid, ax2_grid = np.meshgrid(arrow_x1, arrow_x2)

    thetas = theta * np.ones_like(ax1_grid)
    ux = v * np.cos(thetas)
    uy = v * np.sin(thetas)
    ax.quiver(arrow_x1,arrow_x2,ux,uy,color='k',angles='xy', scale_units='xy', scale=2,alpha=0.5)
    return ax


def plot_pic(env, agent, model_location):

    fig, ([ax1,ax2,ax3,ax4]) = plt.subplots(
        nrows=1, ncols=4,
        figsize=(10.5, 2.5),
        constrained_layout=True,
    )
    
    my_x_ticks = np.arange(-3,3.01,1.5)
    my_y_ticks = np.arange(-3,3.01,1.5)

    labels = ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_xticklabels() + ax2.get_yticklabels() \
    + ax3.get_xticklabels() + ax3.get_yticklabels() + ax4.get_xticklabels() + ax4.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


    '''
    subplot1 : plot the task
    '''
    ax1 = env.plot_task(ax1)
    ax1.set_xticks(my_x_ticks)
    ax1.set_yticks(my_y_ticks)
    ax1.set_xlim((-3, 3))  
    ax1.set_ylim((-3, 3))  
    ax1.tick_params(labelsize=ticks_size)
    ax1.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax1.spines['top'].set_color('gray')      
    ax1.spines['bottom'].set_color('gray')  
    ax1.spines['left'].set_color('gray')   
    ax1.spines['right'].set_color('gray')  

    
    '''
    subplot2,3,4 : plot the feasible region and the learned feasible region for different v and theta
    '''
    ax2 = plot_pr_pic(ax2, agent, v=0.5, theta=np.pi / 4)
    ax2 = env.plot_map(ax2, v=0.5, theta=np.pi / 4)

    ax3 = plot_pr_pic(ax3, agent, v=1, theta=np.pi / 2)
    ax3 = env.plot_map(ax3, v=1, theta=np.pi / 2)

    ax4 = plot_pr_pic(ax4, agent, v=1.5, theta=np.pi / 4, cb =True)
    ax4 = env.plot_map(ax4, v=1.5, theta=np.pi / 4)

    
    for ax in [ax2, ax3, ax4]:
        ax.set_xticks(my_x_ticks)
        ax.set_yticks(my_y_ticks)
        ax.set_xlim((-3, 3))
        ax.set_ylim((-3, 3))
        ax.tick_params(labelsize=ticks_size)
        ax.set_xlim([-2.7,2.7])
        ax.set_ylim([-2.7,2.7])
        ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.spines['bottom'].set_linewidth(width)
        ax.spines['left'].set_linewidth(width)
        ax.spines['right'].set_linewidth(width)
        ax.spines['top'].set_linewidth(width)
        ax.spines['top'].set_color('white') 
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white') 
        ax.spines['right'].set_color('white')
    
    plt.savefig(f"{model_location}/imgs/viz_map.png", dpi=600)


def plot_ctce_value_slice(env, agent, model_location, x_index, y_index, span, grid_size):
    obs, _ = env.reset(seed=int(FLAGS.seed))
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    obs_dim = obs.shape[0]

    if not (0 <= x_index < obs_dim and 0 <= y_index < obs_dim):
        raise ValueError(f'Invalid x_index/y_index for obs_dim={obs_dim}: x_index={x_index}, y_index={y_index}')

    x = np.linspace(-span, span, grid_size)
    y = np.linspace(-span, span, grid_size)
    x_grid, y_grid = np.meshgrid(x, y)
    flat_x = x_grid.ravel()
    flat_y = y_grid.ravel()

    batch_obs = np.repeat(np.expand_dims(obs, axis=0), flat_x.shape[0], axis=0)
    batch_obs[:, x_index] = flat_x
    batch_obs[:, y_index] = flat_y

    safe_value = agent.safe_value.apply_fn({'params': agent.safe_value.params}, jax.device_put(batch_obs))
    value_square = np.asarray(safe_value).reshape(x_grid.shape)

    fig, ax = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
    norm = colors.Normalize(vmin=float(np.min(value_square)), vmax=float(np.max(value_square)))
    ct = ax.contourf(x_grid, y_grid, value_square, norm=norm, levels=30, cmap='viridis')
    ct_line = ax.contour(x_grid, y_grid, value_square, levels=[0], colors='#FFD166', linewidths=1.8)
    ax.clabel(ct_line, inline=True, fontsize=11, fmt='0')
    cb = plt.colorbar(ct, ax=ax, shrink=0.9, pad=0.02)
    cb.ax.tick_params(labelsize=11)

    ax.set_title(f'CTCE Value Slice: obs[{x_index}] vs obs[{y_index}]', fontsize=13)
    ax.set_xlabel(f'obs[{x_index}]', fontsize=12)
    ax.set_ylabel(f'obs[{y_index}]', fontsize=12)

    out_path = os.path.join(model_location, 'imgs', 'viz_map_ctce.png')
    plt.savefig(out_path, dpi=350)
    plt.close(fig)
    return out_path


def load_diffusion_model(model_location):

    with open(os.path.join(model_location, 'config.json'), 'r') as file:
        cfg = to_config_dict(json.load(file))

    env, is_ctce = _make_env(cfg)

    config_dict = dict(cfg['agent_kwargs'])
    model_cls = config_dict.pop("model_cls") 
    agent = globals()[model_cls].create(
        cfg['seed'], env.observation_space, env.action_space, **config_dict
    )

    def get_model_file():
        files = os.listdir(f"{model_location}")
        pickle_files = []
        for file in files:
            if file.endswith('.pickle'):
                pickle_files.append(file)
        numbers = {}
        for file in pickle_files:
            match = re.search(r'\d+', file)
            number = int(match.group())
            path = os.path.join(f"{model_location}", file)
            numbers[number] = path

        max_number = max(numbers.keys())
        max_path = numbers[max_number]
        return max_path
    
    model_file = FLAGS.model_file if FLAGS.model_file != '' else get_model_file()
    new_agent = agent.load(model_file)

    if not os.path.exists(f"{model_location}/imgs"):
        os.makedirs(f"{model_location}/imgs")

    return env, new_agent, cfg, is_ctce

def main(_):
    env, diffusion_agent, cfg, is_ctce = load_diffusion_model(FLAGS.model_location)
    if is_ctce or _infer_num_agents_from_env_name(cfg.get('env_name', '')) is not None:
        out_path = plot_ctce_value_slice(
            env,
            diffusion_agent,
            FLAGS.model_location,
            int(FLAGS.x_index),
            int(FLAGS.y_index),
            float(FLAGS.span),
            int(FLAGS.grid_size),
        )
        print(f'Saved CTCE value slice to: {out_path}')
    else:
        plot_pic(env, diffusion_agent, FLAGS.model_location)
        print(f"Saved PointRobot map to: {os.path.join(FLAGS.model_location, 'imgs', 'viz_map.png')}")
    env.close()


if __name__ == '__main__':
    app.run(main)
