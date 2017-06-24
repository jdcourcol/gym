"""
Simplified F1 game
"""
import random
import logging
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import gym.envs.board_game.race as race
logging.basicConfig(level=logging.DEBUG)


def compute_reward(nb_turn, position, success):
    ''' compute action reward '''
    if not success:
        return -1000 + position
    return 1000 - 2 * nb_turn


class F1Env(gym.Env):
    """
    F1 environment.
    """

    metadata = {"render.modes": ["ansi", "human"]}

    def __init__(self):
        logging.debug('__init__')
        # gear down, same gear, gear up
        self.action_space = spaces.Discrete(3)
        # position health gear curve_stop
        self.observation_space = spaces.Box(low=0, high=400, shape=(1, 9))
        self.circuit = race.CIRCUIT
        self._reset()

    def _seed(self, seed=None):
        logging.debug('_seed')
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        idx = race.get_index_for_position(self.candidate.position,
                                          self.circuit)
        pos_start_idx = 0
        if idx > 0:
            pos_start_idx = race.get_position_for_index(idx - 1, self.circuit)

        remaining = self.candidate.position - pos_start_idx
        is_curve = int(race.is_curve(idx))
        global_remaining = sum(self.circuit[0]) - self.candidate.position
        circuit_seg = 100
        if idx < len(self.circuit[0]):
            circuit_seg = self.circuit[0][idx]
        stop = 2
        if int(idx / 2) < len(self.circuit[1]):
            stop = self.circuit[1][int(idx / 2)]
        return np.array([
            self.candidate.position, self.candidate.health,
            self.candidate.gear, self.candidate.curve_stop, is_curve,
            remaining, global_remaining, circuit_seg, stop
        ])

    def _reset(self):
        logging.debug('_reset')
        self.nb_turn = 0
        self.candidate = race.Candidate(
            position=0, health=10, gear=0, curve_stop=0, done=False)
        return self._get_observation()

    def _step(self, action):
        logging.debug('_step')
        incr_gear = action - 1
        self.candidate.gear += incr_gear
        if self.candidate.gear < 0 or self.candidate.gear > 5:
            return self._get_observation(), -1000, True, {}
        move = random.choice(race.DICE[self.candidate.gear])
        try:
            race.update_candidate(self.candidate, move, self.circuit)
        except:
            return (self._get_observation(), compute_reward(
                self.nb_turn, self.candidate.position, False), True, {})
        self.nb_turn += 1
        if self.candidate.done:
            return (self._get_observation(), compute_reward(
                self.nb_turn, self.candidate.position, True), True, {})
        return self._get_observation(), 0, False, {}

    def _render(self, mode='human', close=False):
        logging.debug('_render')
