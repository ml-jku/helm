from mazelab.generators import random_maze
import numpy as np
from mazelab import BaseMaze
from mazelab import Object
from mazelab import DeepMindColor as color
from mazelab import BaseEnv
from mazelab import VonNeumannMotion
import gym
from gym.spaces import Box
from gym.spaces import Discrete


class Maze(BaseMaze):

    def __init__(self, width=81, height=51, complexity=.25, density=.25, agent_view_size=4):
        x = random_maze(width=width, height=height, complexity=complexity, density=density)
        height, width = x.shape
        new_maze = np.ones((height+2*agent_view_size, width+2*agent_view_size))
        new_maze[agent_view_size:height+agent_view_size, agent_view_size:width+agent_view_size] = x
        self.x = new_maze
        super(Maze, self).__init__()

    @property
    def size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.x == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])
        goal = Object('goal', 3, color.goal, False, [])
        return free, obstacle, agent, goal

    def print(self):
        rows = []
        for row in range(self.x.shape[0]):
            str = np.array(self.x[row], dtype=np.str)
            rows.append(' '.join(str))
        print('\n'.join(rows))


class Env(BaseEnv):
    def __init__(self, complexity=.75, density=.75, agent_view_size=4):
        super().__init__()

        self.complexity = complexity
        self.density = density
        self.agent_view_size = agent_view_size
        # self.maze = Maze(width, height, complexity, density, agent_view_size)
        self.motions = VonNeumannMotion()
        self._sample_env()

        self.observation_space = Box(low=0, high=255, shape=(agent_view_size*2+1, agent_view_size*2+1, 3), dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))


    def _sample_env(self):
        size = np.random.choice(np.arange(5, 26), size=1)[0]
        self.maze = Maze(width=size, height=size, complexity=self.complexity, density=self.density,
                         agent_view_size=self.agent_view_size)
        free_pos = self.maze.objects.free.positions
        start_ind = np.random.choice(len(free_pos))
        self.start_idx = [free_pos[start_ind]]
        self.goal_idx = [[self.agent_view_size + size - 2, self.agent_view_size + size - 2]]

    def _get_pomdp_view(self):
        full_obs = self.get_image()
        y, x = self.maze.objects.agent.positions[0]
        view_size = self.agent_view_size
        partial_obs = full_obs[y-view_size:y+view_size+1, x-view_size:x+view_size+1]
        return partial_obs

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = +1
            done = True
        elif not valid:
            reward = -1
            done = False
        else:
            reward = -0.01
            done = False
        return self._get_pomdp_view(), reward, done, {}

    def reset(self):
        self._sample_env()
        self.maze.objects.agent.positions = self.start_idx
        size = self.maze.size[0]
        self.maze.objects.goal.positions = [[size-2-self.agent_view_size, size-2-self.agent_view_size]]
        return self._get_pomdp_view()

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()


gym.envs.register(id='RandomMaze-v0', entry_point=Env, max_episode_steps=100)


