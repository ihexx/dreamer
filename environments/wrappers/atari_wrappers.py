"""
This code taken from
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

from collections import deque
import os

import cv2
import gym
import gym.spaces as spaces
import numpy as np

os.environ.setdefault("PATH", "")

cv2.ocl.setUseOpenCL(False)


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    # pylint: disable=method-hidden
    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            if not self.vec_env:
                info["TimeLimit.truncated"] = True
            else:
                info.append({"TimeLimit.truncated": True})

        if self.vec_env:
            return observation, reward, done, info
        else:
            o, r, d, i = [], [], [], []
            o.append(observation)
            r.append(reward)
            d.append(done)
            i.append(info)
            return o, r, d, i

    # pylint: disable=method-hidden
    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    # pylint: disable=method-hidden
    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    # pylint: disable=method-hidden
    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    # pylint: disable=method-hidden
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    # pylint: disable=method-hidden
    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    # pylint: disable=method-hidden
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    # pylint: disable=method-hidden
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FrameSkip(gym.Wrapper):
    def __init__(self, env, vec_envs=False, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self.vec_env = vec_envs
        self._skip = skip

    # pylint: disable=method-hidden
    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        if self.vec_env:
            total_reward = [0.0 for _ in range(len(action))]
        done = None

        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if self.vec_env:
                for j in range(len(reward)):
                    total_reward[j] += reward[j]
            else:
                total_reward += reward
            if self.vec_env:
                if any(done):
                    break
            else:
                if done:
                    break

        return obs, total_reward, done, info

    # pylint: disable=method-hidden
    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(
        self, env, vec_envs=False, width=84, height=84, grayscale=True, resize=False
    ):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.vec_env = vec_envs
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.resize = resize

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                height if resize else self.observation_space.shape[0],
                width if resize else self.observation_space.shape[1],
                1 if grayscale else 3,
            ),
            dtype=np.uint8,
        )

    def observation(self, f):
        def convert_1_img(frame):
            if self.resize:
                frame = cv2.resize(
                    frame, (self.width, self.height), interpolation=cv2.INTER_AREA
                )
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = np.expand_dims(frame, -1)
            return frame

        if self.vec_env:
            return [convert_1_img(g) for g in f]
        else:
            return convert_1_img(f)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, vec_envs=False, num_envs=None):
        """Stack k last frames.
        """
        gym.Wrapper.__init__(self, env)
        self.vec_env = vec_envs
        self.num_envs = num_envs
        self.k = k
        if vec_envs:
            self.frames = [deque([], maxlen=k) for _ in range(num_envs)]
        else:
            self.frames = deque([], maxlen=k)

        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype,
        )

    # pylint: disable=method-hidden
    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        if not self.vec_env:
            for _ in range(self.k):
                self.frames.append(ob)
        else:
            for _ in range(self.k):
                [f.append(o) for o, f in zip(ob, self.frames)]
        return self._get_ob()

    # pylint: disable=method-hidden
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        if not self.vec_env:
            self.frames.append(ob)
            return self._get_ob(), reward, done, info
        else:
            [f.append(o) for o, f in zip(ob, self.frames)]
            return self._get_ob(), reward, done, info

    def _get_ob(self):
        if not self.vec_env:
            assert len(self.frames) == self.k
            return LazyFrames(list(self.frames))
        else:
            ret = []
            for f in self.frames:
                assert len(f) == self.k
                ret.append(np.concatenate(f, axis=-1))
            return ret


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env, vec_envs=False):
        gym.ObservationWrapper.__init__(self, env)
        self.vec_env = vec_envs
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        img = np.array(observation, dtype=np.float32)
        img /= 255.0
        return img


class LazyFrames:
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class VectorizedEnvWrapper(gym.Wrapper):
    def __init__(self, make_env, num_envs=1):
        self.envs = [make_env() for env_index in range(num_envs)]
        super().__init__(make_env())

    def reset(self):
        return [env.reset() for env in self.envs]

    def reset_at(self, env_index):
        return self.envs[env_index].reset()

    def step(self, actions):
        next_states, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            next_state, reward, done, info = env.step(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return next_states, rewards, dones, infos

    def render(self, *args, **kwargs):
        return self.envs[0].render(*args, **kwargs)


class Nhwc2Nchw(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """

    def __init__(self, env, vec_envs=False):
        super(Nhwc2Nchw, self).__init__(env)
        self.vec_env = vec_envs
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        if not self.vec_env:
            return np.moveaxis(observation, -1, 0)
        else:
            return [np.moveaxis(o, -1, 0) for o in observation]


def make_atari(env_id, max_episode_steps=None):
    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = FrameSkip(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind(
    env, episode_life=True, clip_rewards=True, frame_stack=4, scale=False
):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, width=96, height=96, resize=True)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack > 0:
        env = FrameStack(env, frame_stack)
    return env


def wrap_pytorch(env):
    return Nhwc2Nchw(env)


def atari_env_generator(env_id, max_episode_steps=None, frame_stack=4, scale=False):
    env = make_atari(env_id, max_episode_steps)
    env = wrap_deepmind(env, frame_stack=frame_stack, scale=scale)
    env = wrap_pytorch(env)
    return env
