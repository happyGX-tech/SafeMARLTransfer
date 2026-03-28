import importlib
import tempfile
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import h5py
import pyrallis
import torch
from tianshou.data import ReplayBuffer, VectorReplayBuffer
from tianshou.data.utils.converter import to_hdf5
from tianshou.env import DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

from fsrl.data import BasicCollector, FastCollector, TrajectoryBuffer
from fsrl.policy import TRPOLagrangian
from fsrl.trainer import OnpolicyTrainer
from fsrl.utils import TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class TrainCfg:
    # unified safety_gym + CTCE config
    task: str = "SafetyAntMultiGoalN3-v0"
    backend: str = "safety_gym"
    num_agents: int = 3
    env_id: Optional[str] = None
    randomize_layout: bool = True
    num_hazards: Optional[int] = None
    world_size: Optional[float] = None

    # schedule / training lengths
    cost_start: float = 15
    cost_end: float = 120
    epoch_start: int = 100
    epoch_end: int = 900
    epoch: int = 1000
    max_traj_len: int = 1500

    # collector behavior
    collect_in_train: bool = True
    collect_only: bool = False
    num_episodes: int = 200
    policy_checkpoint: Optional[str] = None
    policy_device: Optional[str] = None
    dataset_name: str = "offline_dataset_ctce.hdf5"
    dataset_append_mode: str = "shard"  # shard | merge
    output_dir: Optional[str] = None

    # buffer filter
    rmin: float = -9999
    rmax: float = 9999
    cmin: float = 0
    cmax: float = 300

    # runtime
    device: str = "cpu"
    logger_backend: str = "auto"  # auto | tensorboard | wandb
    thread: int = 4
    seed: int = 10

    # TRPOlag model
    lr: float = 5e-4
    hidden_sizes: Optional[Tuple[int, ...]] = None
    unbounded: bool = False
    last_layer_scale: bool = False
    target_kl: float = 0.0005
    backtrack_coeff: float = 0.8
    max_backtracks: int = 10
    optim_critic_iters: int = 20
    gae_lambda: float = 0.95
    norm_adv: bool = True
    use_lagrangian: bool = True
    lagrangian_pid: Tuple[float, ...] = (0.02, 0.0005, 0.02)
    rescaling: bool = True
    gamma: float = 0.99
    max_batchsize: int = 100000
    rew_norm: bool = False
    deterministic_eval: bool = False
    action_scaling: bool = True
    action_bound_method: str = "clip"

    # trainer loop
    episode_per_collect: int = 20
    step_per_epoch: int = 10000 # 用来控制epoch长度，单位是环境交互的step数
    repeat_per_collect: int = 4
    buffer_size: int = 100000
    training_num: int = 20
    testing_num: int = 2
    vector_env: str = "auto"  # auto | dummy | shmem | subproc
    memory_guard: str = "auto"  # auto | on | off
    max_parallel_envs: int = 1
    batch_size: int = 99999
    save_interval: int = 4
    resume: bool = False
    save_ckpt: bool = True
    verbose: bool = True

    # logger outputs
    logdir: str = "logs"
    project: str = "fast-safe-rl"
    group: Optional[str] = None
    name: Optional[str] = None
    prefix: Optional[str] = "trpol"
    suffix: Optional[str] = ""
    strict_checkpoint_shape: bool = True


class ActorProbLargeVar(ActorProb):
    '''Actor with large minimum variance to encourage exploration'''

    SIGMA_MIN = -1
    SIGMA_MAX = 2

    def forward(
        self,
        obs,
        state: Any = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        logits, hidden = self.preprocess(obs, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(
                self.sigma(logits), min=self.SIGMA_MIN, max=self.SIGMA_MAX
            ).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), hidden


def cost_limit_scheduler(epoch, epoch_start, epoch_end, cost_start, cost_end):
    x = min(max(0, epoch - epoch_start), epoch_end - epoch_start)
    cost = cost_start - x * (cost_start - cost_end) / (epoch_end - epoch_start)
    return cost


def _load_checkpoint_config(policy_checkpoint: Optional[str]) -> Dict[str, Any]:
    if not policy_checkpoint:
        return {}
    if os.path.isdir(policy_checkpoint):
        config_path = os.path.join(policy_checkpoint, "config.yaml")
    else:
        parent = os.path.dirname(os.path.dirname(policy_checkpoint))
        config_path = os.path.join(parent, "config.yaml")
    if not os.path.isfile(config_path):
        return {}
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _infer_hidden_sizes_from_checkpoint(policy_checkpoint: Optional[str]) -> Optional[Tuple[int, ...]]:
    cfg = _load_checkpoint_config(policy_checkpoint)
    hidden_sizes = cfg.get("hidden_sizes")
    if not hidden_sizes:
        return None
    return tuple(int(x) for x in hidden_sizes)


def _resolve_hidden_sizes(args: TrainCfg) -> Tuple[int, ...]:
    if args.hidden_sizes:
        return tuple(int(x) for x in args.hidden_sizes)

    ckpt_sizes = _infer_hidden_sizes_from_checkpoint(args.policy_checkpoint)
    if ckpt_sizes:
        return ckpt_sizes

    base = 64 * int(args.num_agents)
    return (base, base)


def _build_env_fn(args: TrainCfg):
    if args.backend != "safety_gym":
        raise ValueError(f"Only backend='safety_gym' is supported, got: {args.backend}")

    env_module = importlib.import_module("envs.fsrl_single_agent_wrapper")
    fsrl_ctce_env = getattr(env_module, "FSRLCTCESingleAgentEnv")

    env_kwargs: Dict[str, Any] = {"randomize_layout": bool(args.randomize_layout)}
    if args.num_hazards is not None:
        env_kwargs["num_hazards"] = int(args.num_hazards)
    if args.world_size is not None:
        env_kwargs["world_size"] = float(args.world_size)

    def _env_fn():
        return fsrl_ctce_env(
            num_agents=int(args.num_agents),
            backend="safety_gym",
            env_id=args.env_id,
            **env_kwargs,
        )

    return _env_fn


def _load_policy_checkpoint(
    policy,
    args: TrainCfg,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    if not args.policy_checkpoint:
        return {}

    checkpoint_obj = None
    if os.path.isdir(args.policy_checkpoint):
        try:
            from fsrl.utils.exp_util import load_config_and_model

            _, checkpoint_obj = load_config_and_model(args.policy_checkpoint, best=False)
        except Exception:
            checkpoint_obj = None
    if checkpoint_obj is None:
        checkpoint_obj = torch.load(args.policy_checkpoint, map_location=args.device)

    if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj:
        state_dict = checkpoint_obj["model"]
    else:
        state_dict = checkpoint_obj

    policy.load_state_dict(state_dict, strict=False)
    if optim is not None and isinstance(checkpoint_obj, dict) and "optimizer" in checkpoint_obj:
        try:
            optim.load_state_dict(checkpoint_obj["optimizer"])
            print("Loaded optimizer state from checkpoint.")
        except Exception as e:
            print(f"[Warn] Failed to load optimizer state: {e}")
    policy.eval()
    return checkpoint_obj if isinstance(checkpoint_obj, dict) else {}


def _validate_checkpoint_compatibility(args: TrainCfg, env) -> None:
    if not args.policy_checkpoint:
        return
    ckpt_cfg = _load_checkpoint_config(args.policy_checkpoint)
    if not ckpt_cfg:
        return

    expected_hidden = tuple(int(x) for x in ckpt_cfg.get("hidden_sizes", []) if x is not None)
    actual_hidden = tuple(int(x) for x in (args.hidden_sizes or ()))
    if expected_hidden and actual_hidden and expected_hidden != actual_hidden:
        msg = (
            f"Checkpoint hidden_sizes mismatch: checkpoint={expected_hidden}, current={actual_hidden}. "
            "Please align hidden_sizes or use checkpoint-compatible config."
        )
        if args.strict_checkpoint_shape:
            raise ValueError(msg)
        print(f"[Warn] {msg}")

    obs_dim_ckpt = ckpt_cfg.get("obs_dim")
    act_dim_ckpt = ckpt_cfg.get("act_dim")
    obs_dim_now = int(env.observation_space.shape[0])
    act_dim_now = int(env.action_space.shape[0])
    if obs_dim_ckpt is not None and int(obs_dim_ckpt) != obs_dim_now:
        msg = f"Checkpoint obs_dim mismatch: checkpoint={obs_dim_ckpt}, current={obs_dim_now}."
        if args.strict_checkpoint_shape:
            raise ValueError(msg)
        print(f"[Warn] {msg}")
    if act_dim_ckpt is not None and int(act_dim_ckpt) != act_dim_now:
        msg = f"Checkpoint act_dim mismatch: checkpoint={act_dim_ckpt}, current={act_dim_now}."
        if args.strict_checkpoint_shape:
            raise ValueError(msg)
        print(f"[Warn] {msg}")


def _resolve_vector_backend(args: TrainCfg) -> str:
    if args.vector_env != "auto":
        return args.vector_env
    return "dummy" if os.name == "nt" else "shmem"


def _apply_memory_guard(args: TrainCfg) -> bool:
    guard_active = False
    if args.memory_guard == "on":
        guard_active = True
    elif args.memory_guard == "auto":
        guard_active = (os.name == "nt" and args.backend == "safety_gym")

    if guard_active:
        limit = max(1, int(args.max_parallel_envs))
        if args.training_num > limit:
            print(f"[Memory Guard] training_num {args.training_num} -> {limit}")
            args.training_num = limit
        if args.testing_num > limit:
            print(f"[Memory Guard] testing_num {args.testing_num} -> {limit}")
            args.testing_num = limit
    return guard_active


def _build_vec_env(env_fns, backend: str):
    if backend == "dummy":
        return DummyVectorEnv(env_fns)
    if backend == "subproc":
        return SubprocVectorEnv(env_fns)
    return ShmemVectorEnv(env_fns)


def _resolve_logger_backend(args: TrainCfg) -> str:
    if args.logger_backend in ("tensorboard", "wandb"):
        return args.logger_backend
    has_wandb_key = bool(os.getenv("WANDB_API_KEY"))
    return "wandb" if has_wandb_key else "tensorboard"


def _resolve_run_dir(args: TrainCfg) -> str:
    return args.output_dir or os.path.join(args.logdir, args.name)


def _default_checkpoint_path(args: TrainCfg) -> str:
    return os.path.join(_resolve_run_dir(args), "checkpoint", "model.pt")


def _copy_hdf5_node(src_node, dst_group, name: str) -> None:
    src_node.file.copy(src_node.name, dst_group, name=name)


def _merge_hdf5_dataset(old_ds, new_ds, dst_group, name: str, chunk_rows: int = 50000) -> None:
    if old_ds.ndim != new_ds.ndim:
        raise ValueError(f"Dataset ndim mismatch at '{name}': {old_ds.ndim} vs {new_ds.ndim}")
    if old_ds.ndim == 0:
        dst_group.create_dataset(name, data=old_ds[()])
        return
    if old_ds.shape[1:] != new_ds.shape[1:]:
        raise ValueError(
            f"Dataset shape mismatch at '{name}': {old_ds.shape} vs {new_ds.shape}"
        )

    old_n = int(old_ds.shape[0])
    new_n = int(new_ds.shape[0])
    total_n = old_n + new_n
    dst_ds = dst_group.create_dataset(
        name,
        shape=(total_n,) + old_ds.shape[1:],
        dtype=old_ds.dtype,
        compression="gzip",
        chunks=True,
    )

    for start in range(0, old_n, chunk_rows):
        end = min(start + chunk_rows, old_n)
        dst_ds[start:end] = old_ds[start:end]

    for start in range(0, new_n, chunk_rows):
        end = min(start + chunk_rows, new_n)
        dst_ds[old_n + start:old_n + end] = new_ds[start:end]


def _merge_hdf5_group(old_group, new_group, dst_group, path_prefix: str = "") -> None:
    keys = sorted(set(old_group.keys()) | set(new_group.keys()))
    for key in keys:
        old_exists = key in old_group
        new_exists = key in new_group
        full_key = f"{path_prefix}/{key}" if path_prefix else key

        if old_exists and not new_exists:
            _copy_hdf5_node(old_group[key], dst_group, key)
            continue
        if new_exists and not old_exists:
            _copy_hdf5_node(new_group[key], dst_group, key)
            continue

        old_node = old_group[key]
        new_node = new_group[key]
        if isinstance(old_node, h5py.Group) and isinstance(new_node, h5py.Group):
            sub_group = dst_group.create_group(key)
            _merge_hdf5_group(old_node, new_node, sub_group, full_key)
            continue
        if isinstance(old_node, h5py.Dataset) and isinstance(new_node, h5py.Dataset):
            _merge_hdf5_dataset(old_node, new_node, dst_group, key)
            continue

        raise ValueError(f"HDF5 node type mismatch at '{full_key}'")


def _append_to_existing_hdf5(dataset_path: str, new_data_batch) -> None:
    dataset_dir = os.path.dirname(dataset_path)
    temp_new_fd, temp_new_path = tempfile.mkstemp(
        prefix="append_new_", suffix=".hdf5", dir=dataset_dir
    )
    os.close(temp_new_fd)
    temp_merged_fd, temp_merged_path = tempfile.mkstemp(
        prefix="append_merged_", suffix=".hdf5", dir=dataset_dir
    )
    os.close(temp_merged_fd)

    try:
        with h5py.File(temp_new_path, "w") as f_new:
            to_hdf5(new_data_batch, f_new, compression="gzip")

        with h5py.File(dataset_path, "r") as f_old, \
             h5py.File(temp_new_path, "r") as f_new, \
             h5py.File(temp_merged_path, "w") as f_merged:
            _merge_hdf5_group(f_old, f_new, f_merged)

        os.replace(temp_merged_path, dataset_path)
    finally:
        if os.path.exists(temp_new_path):
            os.remove(temp_new_path)
        if os.path.exists(temp_merged_path):
            os.remove(temp_merged_path)


def _build_shard_dataset_path(dataset_path: str) -> str:
    stem, ext = os.path.splitext(dataset_path)
    if not ext:
        ext = ".hdf5"
    timestamp = str(int(time.time() * 1000))
    # Use pid to reduce collision risk if multiple runs write to same directory.
    return f"{stem}.part_{os.getpid()}_{timestamp}{ext}"


def _parse_progress_state(progress_path: str) -> Dict[str, float]:
    if not os.path.isfile(progress_path):
        return {}

    with open(progress_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        return {}

    headers = lines[0].split("\t")
    values = lines[-1].split("\t")
    if len(headers) != len(values):
        return {}

    state: Dict[str, float] = {"_epochs": float(len(lines) - 1)}
    for key, value in zip(headers, values):
        try:
            state[key] = float(value)
        except ValueError:
            continue
    return state


def _save_trajectory_buffer(traj_buffer: TrajectoryBuffer, args: TrainCfg) -> str:
    dataset_dir = _resolve_run_dir(args)
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, args.dataset_name)

    if args.resume and os.path.isfile(dataset_path):
        if len(traj_buffer) == 0:
            print(f"No new transitions collected, keep existing dataset: {dataset_path}")
            return dataset_path

        new_batch = traj_buffer.get_all()
        if args.dataset_append_mode == "merge":
            print(f"Appending new transitions to existing dataset: {dataset_path}")
            _append_to_existing_hdf5(dataset_path, new_batch)
            print(f"Finish appending dataset to {dataset_path}!")
            return dataset_path

        shard_path = _build_shard_dataset_path(dataset_path)
        with h5py.File(shard_path, "w") as f:
            to_hdf5(new_batch, f, compression="gzip")
        print(
            "Saved incremental dataset shard (fast mode). "
            f"Base dataset: {dataset_path}, shard: {shard_path}"
        )
        return shard_path

    traj_buffer.save(dataset_dir, dataset_name=args.dataset_name)
    return dataset_path


def _restore_trainer_resume_state(trainer, args: TrainCfg, checkpoint_meta: Dict[str, Any]) -> None:
    if not args.resume:
        return

    restored = False
    trainer_state = checkpoint_meta.get("trainer") if isinstance(checkpoint_meta, dict) else None
    if isinstance(trainer_state, dict):
        trainer.start_epoch = int(trainer_state.get("epoch", 0))
        trainer.best_epoch = trainer.start_epoch
        trainer.env_step = int(trainer_state.get("env_step", 0))
        trainer.cum_episode = int(trainer_state.get("cum_episode", 0))
        trainer.cum_cost = float(trainer_state.get("cum_cost", 0.0))
        restored = True
        print(
            "Resumed trainer state from checkpoint: "
            f"epoch={trainer.start_epoch}, env_step={trainer.env_step}, episodes={trainer.cum_episode}"
        )

    if not restored:
        progress_path = os.path.join(_resolve_run_dir(args), "progress.txt")
        progress_state = _parse_progress_state(progress_path)
        if progress_state:
            trainer.start_epoch = int(progress_state.get("_epochs", 0))
            trainer.best_epoch = trainer.start_epoch
            trainer.env_step = int(progress_state.get("Steps", 0))
            trainer.cum_episode = int(progress_state.get("update/episode", 0))
            trainer.cum_cost = float(progress_state.get("update/cum_cost", 0.0))
            restored = True
            print(
                "Resumed trainer state from progress log: "
                f"epoch={trainer.start_epoch}, env_step={trainer.env_step}, episodes={trainer.cum_episode}"
            )

    if not restored:
        print("[Warn] resume=True but no checkpoint/progress state was found. Starting from epoch 1.")


@pyrallis.wrap()
def train(args: TrainCfg):
    # set seed and computing
    seed_all(args.seed)
    torch.set_num_threads(args.thread)

    default_cfg = TrainCfg()

    guard_active = _apply_memory_guard(args)
    vec_backend = _resolve_vector_backend(args)
    args.hidden_sizes = _resolve_hidden_sizes(args)
    if args.policy_device:
        args.device = args.policy_device

    # logger
    logger_backend = _resolve_logger_backend(args)
    cfg = asdict(args)
    default_cfg = asdict(default_cfg)
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = (
            f"ctce-n{int(args.num_agents)}-cost-"
            f"{int(args.cost_start)}-{int(args.cost_end)}"
        )
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.project, args.group)
    if args.resume and not args.policy_checkpoint:
        auto_ckpt = _default_checkpoint_path(args)
        if os.path.isfile(auto_ckpt):
            args.policy_checkpoint = auto_ckpt
            print(f"Auto resume checkpoint found: {auto_ckpt}")
        else:
            print(f"[Warn] resume=True but default checkpoint not found: {auto_ckpt}")
    cfg = asdict(args)
    if logger_backend == "wandb":
        logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    else:
        logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    print("=" * 72)
    print("TRPOlag Unified Entry")
    print("=" * 72)
    print(f"backend: {args.backend}, num_agents: {args.num_agents}, collect_only: {args.collect_only}")
    print(f"logger_backend: {logger_backend} (configured={args.logger_backend}), vector_env: {vec_backend}")
    print(f"memory_guard: {args.memory_guard} (active={guard_active}, max_parallel_envs={args.max_parallel_envs})")
    print(f"hidden_sizes: {args.hidden_sizes}, device: {args.device}")
    print("=" * 72)

    # model
    env_fn = _build_env_fn(args)
    env = env_fn()
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    net = Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProbLargeVar(
        net,
        action_shape,
        max_action=max_action,
        unbounded=args.unbounded,
        device=args.device
    ).to(args.device)
    critic = [
        Critic(
            Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device),
            device=args.device
        ).to(args.device) for _ in range(2)
    ]
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    actor_critic = ActorCritic(actor, critic)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    if args.last_layer_scale:
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in actor.mu.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = TRPOLagrangian(
        actor,
        critic,
        optim,
        dist,
        logger=logger,
        target_kl=args.target_kl,
        backtrack_coeff=args.backtrack_coeff,
        max_backtracks=args.max_backtracks,
        optim_critic_iters=args.optim_critic_iters,
        gae_lambda=args.gae_lambda,
        advantage_normalization=args.norm_adv,
        use_lagrangian=args.use_lagrangian,
        lagrangian_pid=args.lagrangian_pid,
        cost_limit=args.cost_start,
        rescaling=args.rescaling,
        gamma=args.gamma,
        max_batchsize=args.max_batchsize,
        reward_normalization=args.rew_norm,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_scheduler=None
    )
    checkpoint_meta = _load_policy_checkpoint(policy, args, optim)
    _validate_checkpoint_compatibility(args, env)

    # collector
    traj_buffer = TrajectoryBuffer(
        args.max_traj_len,
        filter_interval=1.5,
        rmin=args.rmin,
        rmax=args.rmax,
        cmin=args.cmin,
        cmax=args.cmax
    )

    if args.collect_only:
        if not args.policy_checkpoint:
            raise ValueError("collect_only=True requires --policy_checkpoint")
        eval_env = env
        collector = BasicCollector(
            policy,
            eval_env,
            ReplayBuffer(1),
            traj_buffer=traj_buffer,
        )
        result = collector.collect(n_episode=args.num_episodes, random=False)
        dataset_path = _save_trajectory_buffer(traj_buffer, args)
        print("Collect-only finished:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print(f"Dataset saved to: {dataset_path}")
        eval_env.close()
        return

    if args.collect_in_train:
        train_collector = BasicCollector(
            policy, env, ReplayBuffer(args.buffer_size), traj_buffer=traj_buffer
        )
    else:
        training_num = min(args.training_num, args.episode_per_collect)
        train_envs = _build_vec_env([env_fn for _ in range(training_num)], vec_backend)
        train_collector = FastCollector(
            policy,
            train_envs,
            VectorReplayBuffer(args.buffer_size, len(train_envs)),
            exploration_noise=True,
        )

    test_collector = BasicCollector(policy, env_fn(), traj_buffer=traj_buffer)

    def stop_fn(reward, cost):
        return False

    trainer = None

    def checkpoint_fn():
        payload = {
            "model": policy.state_dict(),
            "optimizer": optim.state_dict(),
        }
        if trainer is not None:
            payload["trainer"] = {
                "epoch": int(trainer.epoch),
                "env_step": int(trainer.env_step),
                "cum_episode": int(trainer.cum_episode),
                "cum_cost": float(trainer.cum_cost),
            }
        return payload

    if args.save_ckpt:
        logger.setup_checkpoint_fn(checkpoint_fn)

    # trainer
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        batch_size=args.batch_size,
        cost_limit=args.cost_end,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.testing_num,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        logger=logger,
        resume_from_log=args.resume,
        save_model_interval=args.save_interval,
        verbose=args.verbose,
    )
    _restore_trainer_resume_state(trainer, args, checkpoint_meta)

    def saving_dataset():
        dataset_path = _save_trajectory_buffer(traj_buffer, args)
        print(f"Dataset saved to: {dataset_path}")

    def term_handler(signum, frame):
        print("Sig term handler, saving the dataset...")
        if args.save_ckpt:
            logger.save_checkpoint(suffix="interrupt")
        saving_dataset()
        sys.exit(0)

    signal.signal(signal.SIGTERM, term_handler)

    try:
        for epoch, epoch_stat, info in trainer:
            # print(f"Epoch: {epoch}")
            # print(info)
            print(f"Trajs: {len(traj_buffer.buffer)}, transitions: {len(traj_buffer)}")
            cost = cost_limit_scheduler(
                epoch, args.epoch_start, args.epoch_end, args.cost_start, args.cost_end
            )
            policy.update_cost_limit(cost)
            logger.store(tab="train", cost_limit=cost, epoch=epoch)
    except KeyboardInterrupt:
        print("keyboardinterrupt detected, saving the dataset...")
        if args.save_ckpt:
            logger.save_checkpoint(suffix="interrupt")
        saving_dataset()
    except Exception as e:
        print(f"exception catched ({e}), saving the dataset...")
        if args.save_ckpt:
            logger.save_checkpoint(suffix="interrupt")
        saving_dataset()

    finally:
        if not args.collect_in_train:
            try:
                train_envs.close()
            except Exception:
                pass
        try:
            env.close()
        except Exception:
            pass
        try:
            test_collector.env.close()
        except Exception:
            pass

    saving_dataset()


if __name__ == "__main__":
    train()

