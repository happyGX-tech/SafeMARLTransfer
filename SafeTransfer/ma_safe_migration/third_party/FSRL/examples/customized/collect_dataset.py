import importlib
import os
import signal
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import pyrallis
import torch
from tianshou.data import ReplayBuffer, VectorReplayBuffer
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
    target_kl: float = 0.001
    backtrack_coeff: float = 0.8
    max_backtracks: int = 10
    optim_critic_iters: int = 20
    gae_lambda: float = 0.95
    norm_adv: bool = True
    use_lagrangian: bool = True
    lagrangian_pid: Tuple[float, ...] = (0.05, 0.005, 0.1)
    rescaling: bool = True
    gamma: float = 0.99
    max_batchsize: int = 100000
    rew_norm: bool = False
    deterministic_eval: bool = False
    action_scaling: bool = True
    action_bound_method: str = "clip"

    # trainer loop
    episode_per_collect: int = 10
    step_per_epoch: int = 10000
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
        info: Dict[str, Any] = {},
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
        return (mu, sigma), state


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


def _load_policy_checkpoint(policy, args: TrainCfg):
    if not args.policy_checkpoint:
        return

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
    policy.eval()


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


def _save_trajectory_buffer(traj_buffer: TrajectoryBuffer, args: TrainCfg) -> str:
    dataset_dir = args.output_dir or os.path.join(args.logdir, args.name)
    os.makedirs(dataset_dir, exist_ok=True)
    traj_buffer.save(dataset_dir, dataset_name=args.dataset_name)
    return os.path.join(dataset_dir, args.dataset_name)


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
    _load_policy_checkpoint(policy, args)
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

    # def checkpoint_fn():
    #     return {"model": policy.state_dict()}
    # if args.save_ckpt:
    #     logger.setup_checkpoint_fn(checkpoint_fn)

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

    def saving_dataset():
        dataset_path = _save_trajectory_buffer(traj_buffer, args)
        print(f"Dataset saved to: {dataset_path}")

    def term_handler(signum, frame):
        print("Sig term handler, saving the dataset...")
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
        saving_dataset()
    except Exception as e:
        print(f"exception catched ({e}), saving the dataset...")
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

