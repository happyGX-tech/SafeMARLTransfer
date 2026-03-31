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
from matplotlib import patches
import matplotlib.tri as mtri
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
flags.DEFINE_float('boundary_value', float('nan'), 'Optional manual feasible-boundary threshold. NaN means auto by critic_type')
flags.DEFINE_string('output_image_name', '', 'Output image file name or absolute path for visualization result')
flags.DEFINE_bool('semantic_xy', True, 'Use physical x-y semantic visualization for CTCE (recommended).')
flags.DEFINE_integer('semantic_steps', 8000, 'Max rollout steps used to collect semantic x-y samples.')
flags.DEFINE_integer('semantic_episodes', 10, 'Max rollout episodes used to collect semantic x-y samples.')
flags.DEFINE_integer('semantic_agent_index', 0, 'Agent index used for x-y semantic plane in multi-agent env.')
flags.DEFINE_bool('semantic_dual_panel', True, 'Render dual-panel semantic plot: GT panel + learned panel.')


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


def _get_boundary_threshold(cfg):
    if not np.isnan(float(FLAGS.boundary_value)):
        return float(FLAGS.boundary_value), 'manual'

    agent_kwargs = cfg.get('agent_kwargs', {})
    critic_type = str(agent_kwargs.get('critic_type', 'qc')).lower()
    if critic_type == 'hj':
        return 0.0, 'hj_zero_level'

    # qc: feasible typically means expected cumulative cost <= cost_limit.
    cost_limit = float(agent_kwargs.get('cost_limit', 10.0))
    return cost_limit, 'qc_cost_limit'


def _get_value_semantics(cfg):
    critic_type = str(cfg.get('agent_kwargs', {}).get('critic_type', 'qc')).lower()
    if critic_type == 'hj':
        return 'learned feasible value V_h(s)', 'lower = more feasible', critic_type
    return 'learned cost value V_c(s)', 'lower = more feasible', critic_type


def _resolve_output_image_path(model_location, default_name):
    name = (FLAGS.output_image_name or '').strip()
    if name == '':
        name = default_name

    if os.path.isabs(name):
        out_path = name
    else:
        out_path = os.path.join(model_location, 'imgs', name)

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    return out_path


def _extract_semantic_geometry(env, agent_index):
    ma_env = getattr(env, 'ma_env', None)
    if ma_env is None:
        raise ValueError('semantic_xy mode requires CTCE wrapped env with ma_env.')

    base_env = ma_env.unwrapped if hasattr(ma_env, 'unwrapped') else ma_env
    task = getattr(base_env, 'task', None)
    if task is None:
        raise ValueError('Cannot access safety-gym task object for semantic geometry.')

    # Get arena extents from task placement configuration.
    extents = getattr(getattr(task, 'placements_conf', None), 'extents', [-3.0, -3.0, 3.0, 3.0])
    xmin, ymin, xmax, ymax = [float(v) for v in extents]

    # Hazards ground-truth circles.
    hazards_obj = getattr(task, 'hazards', None)
    hazard_radius = float(getattr(hazards_obj, 'size', 0.2)) if hazards_obj is not None else 0.2
    hazard_positions = []
    if hazards_obj is not None and hasattr(hazards_obj, 'pos'):
        for pos in hazards_obj.pos:
            p = np.asarray(pos, dtype=np.float32).reshape(-1)
            if p.shape[0] >= 2:
                hazard_positions.append(p[:2])

    # Optional goals for context.
    goal_positions = []
    if hasattr(task, 'goal_names'):
        for goal_name in getattr(task, 'goal_names'):
            if hasattr(task, goal_name):
                goal_obj = getattr(task, goal_name)
                if hasattr(goal_obj, 'pos'):
                    p = np.asarray(goal_obj.pos, dtype=np.float32).reshape(-1)
                    if p.shape[0] >= 2:
                        goal_positions.append(p[:2])

    # Accessor for agent world position.
    def get_agent_xy():
        return np.asarray(task.agent.pos(int(agent_index))[:2], dtype=np.float32)

    return {
        'extents': (xmin, ymin, xmax, ymax),
        'hazard_positions': hazard_positions,
        'hazard_radius': hazard_radius,
        'goal_positions': goal_positions,
        'get_agent_xy': get_agent_xy,
    }


def plot_ctce_semantic_xy(env, agent, cfg, output_path):
    threshold, threshold_source = _get_boundary_threshold(cfg)
    value_label, direction_note, critic_type = _get_value_semantics(cfg)
    obs, _ = env.reset(seed=int(FLAGS.seed))
    obs = np.asarray(obs, dtype=np.float32)

    geom = _extract_semantic_geometry(env, int(FLAGS.semantic_agent_index))
    xmin, ymin, xmax, ymax = geom['extents']

    xy_points = []
    values = []
    traj_xy = []
    total_steps = 0

    for _ in range(int(FLAGS.semantic_episodes)):
        done = False
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        while not done and total_steps < int(FLAGS.semantic_steps):
            v = agent.safe_value.apply_fn({'params': agent.safe_value.params}, jax.device_put(obs[None, ...]))
            v_scalar = float(np.asarray(v).reshape(-1)[0])
            xy = geom['get_agent_xy']()

            xy_points.append([float(xy[0]), float(xy[1])])
            values.append(v_scalar)
            traj_xy.append([float(xy[0]), float(xy[1])])

            action, agent = agent.eval_actions(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)
            done = bool(terminated or truncated)
            total_steps += 1

        if total_steps >= int(FLAGS.semantic_steps):
            break

    if len(xy_points) < 30:
        raise ValueError('Not enough semantic samples collected for contour plot. Increase --semantic_steps.')

    xy_points = np.asarray(xy_points, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    traj_xy = np.asarray(traj_xy, dtype=np.float32)

    if bool(FLAGS.semantic_dual_panel):
        fig, (ax_gt, ax_learned) = plt.subplots(nrows=1, ncols=2, figsize=(13.6, 6.1), constrained_layout=True)
    else:
        fig, ax_learned = plt.subplots(figsize=(7.8, 6.2), constrained_layout=True)
        ax_gt = None

    tri = mtri.Triangulation(xy_points[:, 0], xy_points[:, 1])

    # Left panel: ground-truth semantic map only.
    if ax_gt is not None:
        for idx, hxy in enumerate(geom['hazard_positions']):
            hazard_fill = patches.Circle((float(hxy[0]), float(hxy[1])), geom['hazard_radius'], facecolor='#EF5350', alpha=0.2, edgecolor='none')
            hazard_edge = patches.Circle((float(hxy[0]), float(hxy[1])), geom['hazard_radius'], facecolor='none', edgecolor='black', linewidth=2.0, linestyle='--')
            ax_gt.add_patch(hazard_fill)
            ax_gt.add_patch(hazard_edge)
            if idx == 0:
                hazard_edge.set_label('GT infeasible (hazard)')

        for idx, gxy in enumerate(geom['goal_positions']):
            ax_gt.scatter(float(gxy[0]), float(gxy[1]), s=45, c='#66BB6A', edgecolors='black', linewidths=0.4, zorder=4)
            if idx == 0:
                ax_gt.text(float(gxy[0]) + 0.06, float(gxy[1]) + 0.06, 'goal', fontsize=9, color='#2E7D32')

        if len(traj_xy) > 2:
            ax_gt.plot(traj_xy[:, 0], traj_xy[:, 1], color='#455A64', alpha=0.3, linewidth=1.1, label='rollout trace')

        ax_gt.set_title(f'Ground-Truth Infeasible Regions\n(agent_{int(FLAGS.semantic_agent_index)})', fontsize=12)
        ax_gt.set_xlabel('world x', fontsize=11)
        ax_gt.set_ylabel('world y', fontsize=11)
        ax_gt.set_xlim([xmin, xmax])
        ax_gt.set_ylim([ymin, ymax])
        ax_gt.set_aspect('equal', adjustable='box')
        ax_gt.grid(alpha=0.2)
        ax_gt.legend(loc='lower left', fontsize=9)

    # Right panel: learned value + learned boundary + GT overlay.
    contourf = ax_learned.tricontourf(tri, values, levels=30, cmap='viridis')

    # Learned feasible boundary.
    has_real_boundary = float(np.min(values - threshold)) <= 0.0 <= float(np.max(values - threshold))
    if has_real_boundary:
        learned_ct = ax_learned.tricontour(tri, values, levels=[threshold], colors='#26C6DA', linewidths=2.3)
        ax_learned.clabel(learned_ct, inline=True, fontsize=10, fmt=f'learned@{threshold:.2f}')
    else:
        proxy_level = float(np.percentile(values, 35.0))
        proxy_ct = ax_learned.tricontour(tri, values, levels=[proxy_level], colors='#26C6DA', linewidths=2.0, linestyles='dashed')
        ax_learned.clabel(proxy_ct, inline=True, fontsize=10, fmt=f'proxy@{proxy_level:.2f}')

    # Ground-truth infeasible regions from hazards.
    for idx, hxy in enumerate(geom['hazard_positions']):
        hazard_fill = patches.Circle((float(hxy[0]), float(hxy[1])), geom['hazard_radius'], facecolor='#EF5350', alpha=0.18, edgecolor='none')
        hazard_edge = patches.Circle((float(hxy[0]), float(hxy[1])), geom['hazard_radius'], facecolor='none', edgecolor='black', linewidth=2.0, linestyle='--')
        ax_learned.add_patch(hazard_fill)
        ax_learned.add_patch(hazard_edge)
        if idx == 0:
            hazard_edge.set_label('GT infeasible (hazard)')

    # Goals for context.
    for idx, gxy in enumerate(geom['goal_positions']):
        ax_learned.scatter(float(gxy[0]), float(gxy[1]), s=42, c='#66BB6A', edgecolors='black', linewidths=0.4, zorder=4)
        if idx == 0:
            ax_learned.text(float(gxy[0]) + 0.06, float(gxy[1]) + 0.06, 'goal', fontsize=9, color='#2E7D32')

    # Trajectory trace.
    if len(traj_xy) > 2:
        ax_learned.plot(traj_xy[:, 0], traj_xy[:, 1], color='#455A64', alpha=0.22, linewidth=1.0, label='rollout trace')

    cb = plt.colorbar(contourf, ax=ax_learned, shrink=0.92, pad=0.02)
    cb.ax.tick_params(labelsize=10)
    cb.set_label(f'{value_label} ({direction_note})', fontsize=10)

    title_suffix = f'boundary={threshold:.2f} ({threshold_source}), {direction_note}'
    if not has_real_boundary:
        title_suffix += ', no real crossing in sampled region'
    ax_learned.set_title(f'Learned Feasible Boundary + GT Overlay\n(agent_{int(FLAGS.semantic_agent_index)}), {title_suffix}', fontsize=12)
    ax_learned.set_xlabel('world x', fontsize=11)
    ax_learned.set_ylabel('world y', fontsize=11)
    ax_learned.set_xlim([xmin, xmax])
    ax_learned.set_ylim([ymin, ymax])
    ax_learned.set_aspect('equal', adjustable='box')
    ax_learned.grid(alpha=0.2)
    ax_learned.legend(loc='lower left', fontsize=9)

    plt.savefig(output_path, dpi=350)
    plt.close(fig)
    return output_path

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
    draw HJ feasible value slice
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
        cb.set_label('learned feasible value V_h(s) (lower = more feasible)', fontsize=12)

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


def plot_pic(env, agent, output_path):

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
    
    plt.savefig(output_path, dpi=600)
    plt.close(fig)


def plot_ctce_value_slice(env, agent, cfg, output_path, x_index, y_index, span, grid_size):
    obs, _ = env.reset(seed=int(FLAGS.seed))
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    obs_dim = obs.shape[0]
    value_label, direction_note, critic_type = _get_value_semantics(cfg)

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

    threshold, threshold_source = _get_boundary_threshold(cfg)
    sign_field = value_square - threshold
    has_real_boundary = float(np.min(sign_field)) <= 0.0 <= float(np.max(sign_field))

    fig, ax = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
    norm = colors.Normalize(vmin=float(np.min(value_square)), vmax=float(np.max(value_square)))
    ct = ax.contourf(x_grid, y_grid, value_square, norm=norm, levels=30, cmap='viridis')

    if has_real_boundary:
        ct_line = ax.contour(
            x_grid,
            y_grid,
            value_square,
            levels=[threshold],
            colors='#2EC4B6',
            linewidths=2.2,
            linestyles='solid',
        )
        ax.clabel(ct_line, inline=True, fontsize=10, fmt=f'feasible@{threshold:.2f}')
    else:
        # If the chosen threshold is outside current value range, draw a proxy contour for readability.
        proxy_level = float(np.percentile(value_square, 35.0))
        proxy_line = ax.contour(
            x_grid,
            y_grid,
            value_square,
            levels=[proxy_level],
            colors='#2EC4B6',
            linewidths=2.2,
            linestyles='dashed',
        )
        ax.clabel(proxy_line, inline=True, fontsize=10, fmt=f'proxy@{proxy_level:.2f}')

    # Shade feasible side for quicker visual parsing.
    if critic_type == 'hj':
        feasible_mask = (value_square <= threshold).astype(np.int32)
    else:
        feasible_mask = (value_square <= threshold).astype(np.int32)
    ax.contourf(
        x_grid,
        y_grid,
        feasible_mask,
        levels=[0.5, 1.5],
        colors=['#2EC4B6'],
        alpha=0.15,
    )

    cb = plt.colorbar(ct, ax=ax, shrink=0.9, pad=0.02)
    cb.ax.tick_params(labelsize=11)
    cb.set_label(f'{value_label} ({direction_note})', fontsize=11)

    boundary_note = f'boundary={threshold:.2f} ({threshold_source}), {direction_note}'
    if not has_real_boundary:
        boundary_note += ', no real crossing in this slice'
    ax.set_title(f'CTCE Value Slice: obs[{x_index}] vs obs[{y_index}]\n{boundary_note}', fontsize=12)
    ax.set_xlabel(f'obs[{x_index}]', fontsize=12)
    ax.set_ylabel(f'obs[{y_index}]', fontsize=12)

    plt.savefig(output_path, dpi=350)
    plt.close(fig)
    return output_path


def load_diffusion_model(model_location):

    if model_location == '':
        raise ValueError(
            '--model_location is required. '
            'You can pass --model_location, or set MODEL_LOCATION, '
            'or run this script inside a directory containing config.json.'
        )

    if not os.path.exists(os.path.join(model_location, 'config.json')):
        raise FileNotFoundError(f'config.json not found in {model_location}')

    with open(os.path.join(model_location, 'config.json'), 'r') as file:
        cfg = to_config_dict(json.load(file))

    env, is_ctce = _make_env(cfg)

    config_dict = dict(cfg['agent_kwargs'])
    model_cls = config_dict.pop("model_cls") 
    # Keep CTCE actor construction consistent with training/eval entrypoints.
    # Some historical config.json files may persist default values
    # (decentralized_actor=False, num_agents=1) even when checkpoints were
    # trained with decentralized actor on multi-agent CTCE tasks.
    if is_ctce or _infer_num_agents_from_env_name(cfg.get('env_name', '')) is not None:
        num_agents = _resolve_ctce_num_agents(cfg)
        if num_agents is None:
            raise ValueError('Cannot resolve num_agents for CTCE model creation in viz_map.')
        config_dict['decentralized_actor'] = True
        config_dict['num_agents'] = int(num_agents)

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
    model_location = _resolve_model_location(FLAGS.model_location)
    env, diffusion_agent, cfg, is_ctce = load_diffusion_model(model_location)
    if is_ctce or _infer_num_agents_from_env_name(cfg.get('env_name', '')) is not None:
        if bool(FLAGS.semantic_xy):
            output_path = _resolve_output_image_path(model_location, 'viz_map_ctce_semantic_xy_dual.png')
            out_path = plot_ctce_semantic_xy(env, diffusion_agent, cfg, output_path)
        else:
            output_path = _resolve_output_image_path(model_location, 'viz_map_ctce.png')
            out_path = plot_ctce_value_slice(
                env,
                diffusion_agent,
                cfg,
                output_path,
                int(FLAGS.x_index),
                int(FLAGS.y_index),
                float(FLAGS.span),
                int(FLAGS.grid_size),
            )
        print(f'Saved CTCE value slice to: {out_path}')
    else:
        output_path = _resolve_output_image_path(model_location, 'viz_map.png')
        plot_pic(env, diffusion_agent, output_path)
        print(f'Saved PointRobot map to: {output_path}')
    env.close()


if __name__ == '__main__':
    app.run(main)
