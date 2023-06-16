import gym
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import numpy as np
from typing import Optional
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize, VecExtractDictObs, VecMonitor, VecTransposeImage
from gym import spaces
from typing import Union, Generator
from queue import Queue
import torch as th
from typing import NamedTuple
from procgen import ProcgenEnv
from envs.random_maze import Env
from variables import _ACTION_MAP, ACTIONS
import dm_env


def generate_in_bg(generator, num_cached=10):
    queue = Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        item = queue.get()


def get_linear_burn_in_fn(start: float, end: float, end_fraction: float, start_fraction: float):

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        elif (1 - progress_remaining) < start_fraction:
            return start
        else:
            return start + (1 - progress_remaining - start_fraction) * (end - start) / (end_fraction - start_fraction)

    return func


def get_exp_decay(start: float, decay: float, start_fraction: float):

    def func(progress_remaining: float, counter: float) -> float:
        if (1 - progress_remaining) < start_fraction:
            return start
        else:
            return start * (decay ** counter)

    return func


def get_exp_weights(n: int):
    weights = np.exp(np.linspace(-1., 0., n))
    weights /= weights.sum()
    return weights


class Minigrid2Image(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces['image']
        self.high = self.observation_space.high[..., 0].flatten()

    def observation(self, observation):
        return observation['image']


class SignObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.high = int(self.env.observation_space.high[0, 0, 0])

    def observation(self, observation):
        return observation['obs']


def make_miniworld_env(id: str):

    def _init():
        import gym_miniworld
        env = gym.make(id)
        if "Sign" in id:
            env = SignObservationWrapper(env)
        return env

    return _init


def make_dmlab_env(id: str):

    def _init():
        import deepmind_lab
        env = DeepMindLabEnvironment(level_name=id)
        return env

    return _init


class DeepMindLabEnvironment(dm_env.Environment):
    """DeepMind Lab environment."""

    def __init__(self, level_name: str, action_repeats: int = 4, record_episodes: bool = False):
        """Construct environment.
        e
        Args:
          level_name: DeepMind lab level name (e.g. 'rooms_watermaze').
          action_repeats: Number of times the same action is repeated on every
            step().
        """
        import deepmind_lab
        config = dict(fps='30',
                      height='80',
                      width='80',
                      maxAltCameraHeight='1',
                      maxAltCameraWidth='1',
                      hasAltCameras='false')

        # seekavoid_arena_01 is not part of dmlab30.
        level_name = 'contributed/dmlab30/{}'.format(level_name)

        self._lab = deepmind_lab.Lab(level_name, ['RGB_INTERLEAVED'], config)
        self._action_repeats = action_repeats
        self._reward = 0
        self._last_done = False
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(80, 80, 3))
        self.action_space = gym.spaces.Discrete(n=len(ACTIONS))
        self.metadata = None
        self.record_episodes = record_episodes
        if record_episodes:
            self.trajectory = []

    def _observation(self):
        last_action = getattr(self, '_action', 0)
        last_reward = getattr(self, '_reward', 0)
        self._last_observation = (
            self._lab.observations()['RGB_INTERLEAVED'],
            np.array(last_action, dtype=np.int64),
            np.array(last_reward, dtype=np.float32)
        )
        return self._last_observation

    def reset(self):
        self._lab.reset()
        self.trajectory = []
        stepdata = dm_env.restart(self._observation())
        return stepdata.observation[0]

    def step(self, action):
        if not self._lab.is_running():
            # Dump collected episode
            stepdata = dm_env.restart(self.reset())
            self.trajectory.append(stepdata)
            return stepdata.observation, self._reward, True, {}

        self._action = action.item()
        if self._action not in _ACTION_MAP:
            raise ValueError('Action not available')
        lab_action = np.array(_ACTION_MAP[self._action], dtype=np.intc)
        self._reward = self._lab.step(lab_action, num_steps=self._action_repeats)

        if self._lab.is_running():
            stepdata = dm_env.transition(self._reward, self._observation())
            self.trajectory.append(stepdata)
            return stepdata.observation[0], self._reward, False, {}

        stepdata = dm_env.termination(self._reward, self._last_observation)
        self.trajectory.append(stepdata)
        return stepdata.observation[0], self._reward, True, {}

    def observation_spec(self):
        return (
                    dm_env.specs.Array(shape=(80, 80, 3), dtype=np.uint8),
                    dm_env.specs.Array(shape=(), dtype=np.int64),
                    dm_env.specs.Array(shape=(), dtype=np.float32)
        )

    def action_spec(self):
        return dm_env.specs.DiscreteArray(num_values=15, dtype=np.int64)

    def seed(self, seed):
        pass



def make_minigrid_env(id: str):
    """
    Utility function for multiprocessed env.

    :param id: (str) the environment ID
    """

    def _init():
        env = gym.make(id)
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        return env

    return _init


def make_maze_env():
    def _init():
        env = gym.make('RandomMaze-v0')
        return env

    return _init


def make_procgen_env(id, num_levels, num_envs):
    env = ProcgenEnv(env_name=id, num_envs=num_envs, distribution_mode='memory', num_levels=num_levels,
                     start_level=0)
    env = VecExtractDictObs(env, 'rgb')
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=1.)
    return env


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    hiddens: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class RecurrentRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    hiddens: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    valid_mask: th.Tensor


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        hidden_dim: int = 1024
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.dones = None
        self.generator_ready = False
        self.hidden_dim = hidden_dim
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        if self.hidden_dim:
            self.hiddens = np.zeros((self.buffer_size, self.n_envs, self.hidden_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(self, last_values: np.ndarray, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).

        """
        # Convert to numpy
        last_values = last_values.copy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        hidden: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = np.array(log_prob).reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        if self.hidden_dim:
            self.hiddens[self.pos] = np.array(hidden).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = np.array(value).copy()
        self.log_probs[self.pos] = np.array(log_prob).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, n_batches: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        batch_size = self.buffer_size * self.n_envs // n_batches
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]
            if self.hidden_dim:
                _tensor_names.append("hiddens")

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])

            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None):
        hiddens = self.hiddens[batch_inds] if self.hidden_dim else np.zeros((0,))
        data = (
            self.observations[batch_inds],
            hiddens,
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        data = RolloutBufferSamples(*tuple(map(self.to_torch, data)))
        return data


class RecurrentRolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        hidden_dim: int = 256
    ):
        super(RecurrentRolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.dones = None
        self.generator_ready = False
        self.hidden_dim = hidden_dim
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.hiddens = np.zeros((self.buffer_size, self.n_envs, 2, self.hidden_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RecurrentRolloutBuffer, self).reset()

    def compute_returns_and_advantage(self, last_values: np.ndarray, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).

        """
        # Convert to numpy
        last_values = last_values.copy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        hidden: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = np.array(log_prob).reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.hiddens[self.pos] = np.array(hidden).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = np.array(value).copy()
        self.log_probs[self.pos] = np.array(log_prob).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RecurrentRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.hiddens[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds],
            self.log_probs[batch_inds],
            self.advantages[batch_inds],
            self.returns[batch_inds],
            self.valid_masks[batch_inds]
        )
        data = RecurrentRolloutBufferSamples(*tuple(map(self.to_torch, data)))
        return data

    def _pad_sequences(self, seqs, maxlen):
        padded_seqs = []
        masks = []
        for seq in seqs:
            valid_mask = np.ones((maxlen,))
            diff = maxlen - len(seq)
            padded_seqs.append(np.concatenate((seq, np.zeros((diff, *seq.shape[1:]), dtype=np.float32))))
            valid_mask[len(seq):] = 0
            masks.append(valid_mask)
        return np.array(padded_seqs), np.array(masks, dtype=np.int32)

    def get(self, batch_size: Optional[int] = None):
        assert self.full, ""
        # Prepare the data
        if not self.generator_ready:
            self._prepare_seqs()

            self.generator_ready = True

        start_idx = 0
        indices = np.random.permutation(len(self.observations))
        while start_idx < len(indices):
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _prepare_seqs(self):
        _tensor_names = [
            "observations",
            "hiddens",
            "actions",
            "values",
            "log_probs",
            "advantages",
            "returns"
        ]
        tensor_dict = self.__dict__
        for t in _tensor_names:
            subseqs = []
            seqlens = []
            for i in range(self.n_envs):
                if tensor_dict['episode_starts'][:, i].any():
                    ep_starts = np.nonzero(tensor_dict['episode_starts'][:, i])[0]
                else:
                    ep_starts = [0, self.buffer_size]
                if ep_starts[0] != 0:
                    ep_starts = np.insert(ep_starts, 0, 0)
                if ep_starts[-1] != self.buffer_size:
                    ep_starts = np.insert(ep_starts, len(ep_starts), self.buffer_size)
                for start, end in zip(ep_starts[:-1], ep_starts[1:]):
                    subseqs.append(tensor_dict[t][start: end, i, ...])
                    seqlens.append(end-start)
            self.__dict__[t] = subseqs

        maxlen = np.max(seqlens)
        for t in _tensor_names:
            self.__dict__[t], self.valid_masks = self._pad_sequences(self.__dict__[t], maxlen)
