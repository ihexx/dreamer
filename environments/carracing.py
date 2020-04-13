"""

"""
from environments.wrappers import ObsDict, FrameSkip, WarpFrame, NormalizeActions, Wrapper
import gym
import numpy as np


class CarRacing(Wrapper):
    def __init__(self, action_repeat=10, size=(84, 84), grayscale=True):
        env = gym.make("CarRacing-v0", verbose=0)
        env = NormalizeActions(env)
        env = _HelperWrapperCarRacing(env, punish_grass=True)
        env = FrameSkip(env, skip=action_repeat)
        env = WarpFrame(env, grayscale=grayscale, width=size[0], height=size[1], resize=True)
        env = ObsDict(env, "image")
        Wrapper.__init__(self, env)
        # env = atari_wrappers.ScaledFloatFrame(env)  # /255
        # env = atari_wrappers.FrameStack(env, args.frame_stack)
        # env = atari_wrappers.Nhwc2Nchw(env)


class _HelperWrapperCarRacing(gym.Wrapper):
    def __init__(self, env: gym.Env, punish_grass=False):
        """
        - Adds an action mapping so all actions go from range [-1:1], and get mapped to the environment's actual limits
        this way you can just set a tanh() activation on your network and forget action space limits
        - Adds negative reward for grass (checks that screen is full on green)
        - Kills the episode if agent gets no rewards for a while

        Disable `tanh_actions` when running in discrete mode
        :param env:
        :param punish_grass:
        # """
        gym.Wrapper.__init__(self, env)
        self.info_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,))
        self.info_state_keys = ("speed",)
        self.punish_grass = punish_grass

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

    def reset(self, **kwargs):
        self.av_r = self.reward_memory()
        return gym.Wrapper.reset(self, **kwargs)

    def step(self, action):
        next_state, reward, done, info = gym.Wrapper.step(self, action)

        true_speed = np.sqrt(
            np.square(self.unwrapped.car.hull.linearVelocity[0])
            + np.square(self.unwrapped.car.hull.linearVelocity[1])
        )
        info["speed"] = true_speed
        if self.punish_grass:
            if np.mean(next_state[:, :, 1]) > 185.0:
                reward -= 0.05
        done |= self.av_r(reward) <= -0.1
        return next_state, reward, done, info
