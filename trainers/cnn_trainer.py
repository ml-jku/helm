import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
import os
import random
import zipfile
import glob
import json
import torch.cuda
from gym import spaces
from torch.nn import functional as F
import time
from utils import get_linear_burn_in_fn, get_exp_decay, RolloutBuffer
from variables import procgen_envs
from model import HELM, LSTMImpalaAgent, MarkovianImpalaCNN

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, constant_fn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean, get_latest_run_id, get_linear_fn
from stable_baselines3.common.running_mean_std import RunningMeanStd


class CNNPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        lr_decay: str = "none",
        ent_decay: str = "none",
        ent_decay_factor: float = 0.9,
        n_envs: int = 1,
        start_fraction: float = 0,
        end_fraction: float = 1.0,
        min_lr: float = 1e-7,
        min_ent_coef: float = 1e-4,
        config: dict = None,
        clip_decay: int = None,
        adv_norm: bool = False,
        save_ckpt: bool = True
    ):

        super(CNNPPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a multiple of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )

        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.highest_return = 0.0
        self.config = config

        if lr_decay == 'none':
            self.learning_rate = constant_fn(learning_rate)
        elif lr_decay == 'linear':
            self.learning_rate = get_linear_fn(learning_rate, min_lr, end_fraction=end_fraction)
        elif lr_decay == 'linear_burn_in':
            self.learning_rate = get_linear_burn_in_fn(learning_rate, min_lr, start_fraction=start_fraction, end_fraction=end_fraction)
        elif lr_decay == 'exp':
            self.learning_rate = get_exp_decay(learning_rate, ent_decay_factor, start_fraction=start_fraction)
        else:
            raise NotImplementedError(f"Learning Rate Decay {lr_decay} not implemented")

        if ent_decay == 'none':
            self.ent_coef = constant_fn(ent_coef)
        elif ent_decay == 'linear':
            self.ent_coef = get_linear_fn(ent_coef, min_ent_coef, end_fraction=end_fraction)
        elif ent_decay == 'linear_burn_in':
            self.ent_coef = get_linear_burn_in_fn(ent_coef, min_ent_coef, start_fraction=start_fraction, end_fraction=end_fraction)
        elif ent_decay == 'exp':
            self.ent_coef = get_exp_decay(ent_coef, ent_decay_factor, start_fraction)
        else:
            raise NotImplementedError(f"Entropy Decay {ent_decay} not implemented")

        if clip_decay == 'none':
            self.clip_range = constant_fn(clip_range)
        elif clip_decay == 'linear':
            self.clip_range = get_linear_fn(clip_range, end=0, end_fraction=1)
        else:
            raise NotImplementedError(f"Clipping factor decay {clip_decay} not implemented")

        self.entropy_coef = ent_coef
        self.lr_decay = lr_decay
        self.ent_decay = ent_decay
        self.counter = 0
        self.n_envs = n_envs
        self.adv_norm = adv_norm
        self.save_ckpt = save_ckpt
        if self.adv_norm:
            self._adv_rms = RunningMeanStd(shape=())

        if _init_setup_model:
            self._setup_model()

        if seed is not None:
            self._set_seed(seed)

        self.rollout_buffer = RolloutBuffer(self.n_steps, self.observation_space, self.action_space, device,
                                            gamma=gamma, gae_lambda=gae_lambda, n_envs=n_envs,
                                            hidden_dim=0)

        self.policy = MarkovianImpalaCNN(self.observation_space.shape, env.action_space.n, self.config['optimizer'],
                                         self.config['learning_rate']).to(self.device)

    def _set_seed(self, seed: int) -> None:
        """
        Seed the different random generators.

        :param seed:
        """
        # Seed python RNG
        random.seed(seed)
        # Seed numpy RNG
        np.random.seed(seed)
        # seed the RNG for all devices (both CPU and CUDA)
        th.manual_seed(seed)
        # Deterministic operations for CuDNN, it may impact performances
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        self.action_space.seed(seed)
        if self.env is not None and self.config['env'] not in procgen_envs:
            # procgen environments do not support setting the seed that way
            self.env.seed(seed)

    def _setup_model(self) -> None:
        super(CNNPPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        # self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        self.ent_coef_schedule = get_schedule_fn(self.ent_coef)
        self.clip_range_schedule = get_schedule_fn(self.clip_range)

    def _update_entropy_coef(self) -> None:
        # Log the current entropy coefficient
        if self.ent_decay == 'exp':
            self.ent_coef = self.ent_coef_schedule(self._current_progress_remaining, self.counter)
        else:
            self.ent_coef = self.ent_coef_schedule(self._current_progress_remaining)
        self.logger.record("train/ent_coef", self.ent_coef)

    def _dump_sources(self, outpath) -> None:
        zipf = zipfile.ZipFile(os.path.join(outpath, 'LMRL.zip'), 'w', zipfile.ZIP_DEFLATED)
        src_files = glob.glob(f'{os.path.abspath(".")}/**/*.py', recursive=True)
        for file in src_files:
            zipf.write(file, os.path.relpath(file, '../'))

    def _dump_config(self, outpath) -> None:
        with open(os.path.join(outpath, '../config.json'), 'w') as f:
            f.write(json.dumps(self.config, indent=4, sort_keys=True))

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        self._update_entropy_coef()
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        else:
            clip_range_vf = None

        self.policy.train()

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        clip_fractions = []
        exp_vars = []
        approx_kl_divs = []
        losses = []
        # train for n_epochs epochs
        start = time.time()
        for epoch in range(self.n_epochs):

            generator = self.rollout_buffer.get(n_batches=self.config['n_batches'])

            for rollout_data in generator:

                actions = rollout_data.actions
                advantages = rollout_data.advantages
                old_values = rollout_data.old_values
                observations = rollout_data.observations
                old_log_prob = rollout_data.old_log_prob
                returns = rollout_data.returns

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long()

                values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)

                # Normalize advantage
                if len(advantages) > 1:
                    values = values.squeeze()
                    self.logger.record('train/adv_std', advantages.std().detach().cpu().item())
                    self.logger.record('train/adv_mean', advantages.mean().detach().cpu().item())
                    if self.adv_norm:
                        # use running stats for advantage normalization
                        self._adv_rms.update(advantages.detach().cpu().numpy())
                        advantages = self._norm_advs(advantages)
                    else:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - old_log_prob)

                if isinstance(self.action_space, spaces.Box):
                    advantages = advantages.unsqueeze(-1)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                # Logging

                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = old_values + th.clamp(values - old_values, -clip_range_vf, clip_range_vf)
                # Value loss using the TD(gae_lambda) target
                # Normalize returns
                value_loss = F.mse_loss(returns, values_pred)

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                entropy_losses.append(entropy_loss.item())
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                clip_fractions.append(clip_fraction)

                # Re-sample the noise matrix because the log_std has changed
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.config['batch_size'])

                # Optimization step
                self.policy.optimizer.zero_grad(set_to_none=True)
                if hasattr(self.policy, 'trxl_optimizer'):
                    self.policy.trxl_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                approx_kl_divs.append(th.mean(old_log_prob - log_prob).detach().cpu().numpy())

                exp_var = explained_variance(values.cpu().detach().numpy().flatten(),
                                             returns.cpu().detach().numpy().flatten())
                exp_vars.append(exp_var)
                losses.append(loss.detach().cpu().item())

            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

            self._n_updates += 1

        explained_var = np.mean(exp_vars)
        end = time.time()
        print(f"Update Time: {end - start} seconds")

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _norm_advs(self, advs: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        advs = advs / np.sqrt(self._adv_rms.var + epsilon)
        return advs

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer,
                         n_rollout_steps: int) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        start = time.time()
        n_steps = 0

        rollout_buffer.reset()
        self.policy.eval()

        callback.on_rollout_start()

        start = time.time()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                image_obs = self._last_obs
                high = env.observation_space.high.reshape(-1)[0]
                observations = torch.tensor(image_obs / high).float().to(self.device)
                action, value, log_prob = self.policy(observations)
                hidden = None

            new_obs, rewards, dones, infos = env.step(action)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            n_steps += 1
            self._update_info_buffer(infos)

            add_obs = observations.cpu().numpy()
            action = np.expand_dims(action, axis=-1)
            rollout_buffer.add(add_obs, np.array([]), action, rewards, self._last_episode_starts, value, log_prob)
            self._last_obs = new_obs
            self._last_episode_starts = dones


        with th.no_grad():
            image_obs = self._last_obs
            high = env.observation_space.high.reshape(-1)[0]
            observations = torch.tensor(image_obs / high).float().to(self.device)
            action, value, log_prob = self.policy(observations)

        rollout_buffer.compute_returns_and_advantage(last_values=value, dones=self._last_episode_starts)
        callback.on_rollout_end()

        if self.save_ckpt:
            if len(self.ep_info_buffer) < self.n_envs:
                ret = [ep_info['r'] for ep_info in self.ep_info_buffer]
                ret = safe_mean(ret + [0.] * (self.n_envs - len(ret)))
            else:
                ret = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
            if not self.counter or self.highest_return < ret:
                self.highest_return = ret if not np.isnan(ret) else 0.0
                checkpoint = self._prepare_checkpoint()
                th.save(checkpoint, os.path.join(self.save_path, f'ckpt_best.pt'))
                print("Saved model checkpoint!!")

        end = time.time()
        print(f"Rollout Time: {end - start}\n")

        return True

    def _prepare_checkpoint(self):
        ent_coef = self.ent_coef_schedule(self._current_progress_remaining, self.counter) \
            if self.ent_decay == 'exp' else self.ent_coef_schedule(self._current_progress_remaining)
        lr = self.lr_schedule(self._current_progress_remaining)
        clip_range = self.clip_range_schedule(self._current_progress_remaining)
        module_names = [d for d in dir(self.policy) if not d.startswith('_') and
                        isinstance(getattr(self.policy, d), torch.nn.Module) and d != 'model']
        checkpoint = {
            'network': {},
            'optimizer': self.policy.optimizer.state_dict(),
            'num_timesteps': self.num_timesteps,
            'ent_coef': ent_coef,
            'learning_rate': lr,
            'clip_range': clip_range
        }
        for m_name in module_names:
            checkpoint['network'][m_name] = getattr(self.policy, m_name).state_dict()
        checkpoint['seed'] = (np.random.get_state(), torch.get_rng_state(), random.getstate())
        return checkpoint

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "PPO",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, None, callback, 1, 100, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        latest_run = get_latest_run_id(self.tensorboard_log, 'PPO')
        self.save_path = os.path.join(os.path.join(self.tensorboard_log, f'PPO_{latest_run}'))
        callback.on_training_start(locals(), globals())
        self._dump_sources(self.save_path)
        self._dump_config(self.save_path)

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    returns = [ep_info["r"] for ep_info in self.ep_info_buffer]
                    if len(returns) < self.n_envs:
                        returns = returns + [0] * (self.n_envs - len(returns))
                    self.logger.record("rollout/ep_rew_mean", safe_mean(returns))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()
            self.counter += 1

        callback.on_training_end()

        return self
