import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import logging

logger = logging.getLogger(__name__)


class BitSwapEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n, explicit_goal, fixed_goal):
        super(BitSwapEnvironment, self).__init__()
        self.n = n
        self.explicit_goal = explicit_goal
        self.fixed_goal = fixed_goal
        self.action_space = spaces.Discrete(self.n)
        self.observation_space = spaces.MultiBinary(self.n)
        self.max_episode_steps = n
        self.steps = 0
        self.state = None
        self.goal = np.random.randint(2, size=self.n)
        self.seed = self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _calc_reward(self):
        return int(all(self.state == self.goal))

    def step(self, action):
        self.state[action] = not self.state[action]
        obs = self._get_obs()
        reward = self._calc_reward()
        terminal = True if reward == 1 else False
        self.steps += 1
        if self.steps == self.max_episode_steps:
            terminal = True
        return obs, reward, terminal, None


    def _get_obs(self):
        ret = {}
        ret['observation'] = self.state.copy()
        if self.explicit_goal:
            ret['desired_goal'] = self.goal.copy()
            ret['achieved_goal'] = self.state.copy()
        else:
            ret['desired_goal'] = None
            ret['achieved_goal'] = None
        return ret


    def reset(self):
        self.steps = 0
        self.state = np.random.randint(2, size=self.n)
        if not self.fixed_goal:
            self.goal = np.random.randint(2, size=self.n)
        return self._get_obs()

    def render(self, mode='human', close=False):
        logger.info('\n State: %s \n Goal:  %s' % (self.state, self.goal))
