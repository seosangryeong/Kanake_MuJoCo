"""IsaacLab-derived PPO settings and portable MuJoCo checkpoints."""

from copy import deepcopy
from dataclasses import asdict

import torch
from rsl_rl.runners import OnPolicyRunner

CHECKPOINT_VERSION = 4  # Raised-head mode adds the CEM neck pose prior.
COMPATIBLE_VERSIONS = (2, 3, 4)


def runner_config(rollout):
    return {
        "num_steps_per_env": rollout,
        "save_interval": 50,
        "empirical_normalization": True,
        "logger": "tensorboard",
        "policy": {
            "class_name": "ActorCritic", "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128], "activation": "elu",
            "init_noise_std": 0.5, "noise_std_type": "log",
        },
        "algorithm": {
            "class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True,
            "clip_param": 0.2, "entropy_coef": 0.005, "num_learning_epochs": 5,
            "num_mini_batches": 4, "learning_rate": 0.001, "schedule": "adaptive",
            "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0,
        },
    }


class KanakeRunner(OnPolicyRunner):
    def __init__(self, env, config, env_config, log_dir=None, device="cpu"):
        self.kanake_config = {
            "version": CHECKPOINT_VERSION, "environment": asdict(env_config), "ppo": config,
        }
        super().__init__(env, deepcopy(config), log_dir=log_dir, device=device)
        self.git_status_repos = []

    def save(self, path, infos=None):
        super().save(path, infos={**(infos or {}), "kanake": self.kanake_config})

    def restore(self, state, *, resume=False):
        self.alg.policy.load_state_dict(state["model_state_dict"])
        self.obs_normalizer.load_state_dict(state["obs_norm_state_dict"])
        self.privileged_obs_normalizer.load_state_dict(state["privileged_obs_norm_state_dict"])
        if resume:
            self.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
            self.current_learning_iteration = state["iter"] + 1


def warm_start(runner, state, old_env, new_env):
    """Start a policy from an older checkpoint whose observations or actions were a subset.

    New observation columns are appended after the old ones, so the old first-layer weights are
    copied and the new columns start at zero. Actions are remapped (v9 per-joint amplitudes all
    start from the old shared amplitude) and residuals whose range changed are rescaled, so the
    warm-started policy initially commands the same joint targets as the old one.
    """
    from kanake_mujoco.config import EnvConfig, GAIT_SIZE, WAVE_JOINTS, action_size
    from kanake_mujoco.env import residual_layout

    old_cfg = EnvConfig(**old_env)
    policy = runner.alg.policy
    old = state["model_state_dict"]
    new = policy.state_dict()
    old_obs, new_obs = old["actor.0.weight"].shape[1], new["actor.0.weight"].shape[1]
    old_critic = old["critic.0.weight"].shape[1]
    # The critic keeps its old privileged columns only if the policy observation did not grow.
    critic_width = old_critic if new_obs == old_obs else old_obs

    old_gait = 2 + WAVE_JOINTS if old_cfg.amplitude_per_joint else GAIT_SIZE
    new_gait = 2 + WAVE_JOINTS if new_env.amplitude_per_joint else GAIT_SIZE
    if new_env.amplitude_per_joint and not old_cfg.amplitude_per_joint:
        rows = [1, 2] + [0] * WAVE_JOINTS  # frequency, steering, then the shared amplitude n times
    elif new_env.amplitude_per_joint == old_cfg.amplitude_per_joint:
        rows = list(range(new_gait))
    else:
        raise ValueError("Cannot warm-start shared amplitudes from per-joint amplitudes")
    rows += list(range(old_gait, old_gait + 16))
    assert len(rows) == action_size(new_env)

    for key, value in old.items():
        target = new[key]
        if key == "actor.0.weight":
            target.zero_()
            target[:, :old_obs] = value
        elif key == "critic.0.weight":
            target.zero_()
            target[:, :critic_width] = value[:, :critic_width]
        elif key in ("actor.6.weight", "actor.6.bias", "log_std"):
            target.copy_(value[rows])
        elif value.shape == target.shape:
            target.copy_(value)
        else:
            raise ValueError(f"Cannot warm-start {key}: {tuple(value.shape)} → {tuple(target.shape)}")
    old_scale = residual_layout(old_cfg.neck_joint, old_cfg.head_lift, old_cfg.climb, old_cfg.climb_pitch,
                                old_cfg.climb_yaw)[0]
    new_scale = residual_layout(new_env.neck_joint, new_env.head_lift, new_env.climb, new_env.climb_pitch,
                                new_env.climb_yaw)[0]
    ratio = torch.as_tensor(old_scale / new_scale, dtype=torch.float32, device=new["log_std"].device)
    residual = slice(new_gait, new_gait + len(ratio))
    new["actor.6.weight"][residual] *= ratio[:, None]
    new["actor.6.bias"][residual] *= ratio
    new["log_std"][residual] += torch.log(ratio)
    policy.load_state_dict(new)

    def extend(normalizer, saved, width):
        for name in ("_mean", "_var", "_std"):
            buffer = getattr(normalizer, name)
            buffer[:, :width] = saved[name][:, :width].to(buffer.device)
        # A modest count lets the new columns' statistics adapt within a few iterations.
        normalizer.count.fill_(min(float(saved["count"]), 2e5))

    extend(runner.obs_normalizer, state["obs_norm_state_dict"], old_obs)
    if not isinstance(runner.privileged_obs_normalizer, torch.nn.Identity):
        # Without a privileged critic, the old critic read the normalized policy observation.
        saved = state["privileged_obs_norm_state_dict"] if old_critic > old_obs else state["obs_norm_state_dict"]
        extend(runner.privileged_obs_normalizer, saved, critic_width)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    metadata = (state.get("infos") or {}).get("kanake", {})
    if metadata.get("version") not in COMPATIBLE_VERSIONS:
        raise ValueError("Expected a serpentine Kanake MuJoCo checkpoint from rl/train.py")
    if metadata["version"] == 3 and metadata["environment"].get("head_lift"):
        raise ValueError("Raised-head checkpoints before version 4 lack the neck pose prior")
    if metadata["version"] == 2:
        # Version 2 drove j2 with the wave and filtered every residual at 80 ms.
        metadata["environment"] = {**metadata["environment"], "neck_joint": False}
        metadata["version"] = CHECKPOINT_VERSION
    return state, metadata


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable to PyTorch")
    return requested
