"""
Simplified F1 game
"""
import random
import logging
import copy
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import race
logging.basicConfig(level=logging.DEBUG)


def compute_reward(nb_turn, position, success):
    ''' compute action reward '''
    if not success:
        return -1000 + position
    return 10000 - 50 * nb_turn


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
        self.observation_space = spaces.Box(low=0, high=400, shape=(1, 11))
        self.circuit = race.CIRCUIT
        self.log_file_name = None
        self.all_log = []
        self._reset()

    def _seed(self, seed=None):
        logging.debug('_seed')
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        idx = race.get_index_for_position(self.candidate.position,
                                          self.circuit)
        remaining = 0
        if idx < len(self.circuit[0]):
            pos_start_idx = race.get_position_for_index(idx + 1, self.circuit)
            remaining = pos_start_idx - self.candidate.position
        is_curve = int(race.is_curve(idx))
        global_remaining = sum(self.circuit[0]) - self.candidate.position
        circuit_seg = 100
        circuit_seg_1 = 100
        if idx < len(self.circuit[0]):
            circuit_seg = self.circuit[0][idx]
        if (idx + 1) < len(self.circuit[0]):
            circuit_seg_1 = self.circuit[0][idx + 1]
        stop = 2
        stop_1 = 2
        if int(idx / 2) < len(self.circuit[1]):
            stop = self.circuit[1][int(idx / 2)]
        if int((idx + 1) / 2) < len(self.circuit[1]):
            stop_1 = self.circuit[1][int((idx + 1) / 2)]

        return np.array([
            self.candidate.position, self.candidate.health,
            self.candidate.gear, self.candidate.curve_stop, is_curve,
            remaining, global_remaining, circuit_seg, stop, circuit_seg_1,
            stop_1
        ])

    def _reset(self):
        logging.debug('_reset')
        self.nb_turn = 0
        self.cur_log = []
        self.all_log.append(self.cur_log)
        self.candidate = race.Candidate(
            position=0, health=10, gear=0, curve_stop=0, done=False)
        return self._get_observation()

    def _step(self, action):
        logging.debug('_step')
        if self.log_file_name:
            self.cur_log.append(copy.copy(self._get_observation()))
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
        return self._get_observation(
        ), move + 20 * self.candidate.gear, False, {}

    def _render(self, mode='human', close=False):
        logging.debug('_render')

    def _close(self):
        if self.log_file_name:
            import pandas as pd
            import numpy as np
            l = [
                np.append(c, idx)
                for idx, log in enumerate(self.all_log) for c in log
            ]
            df = pd.DataFrame(
                l,
                columns=[
                    'position', 'health', 'gear', 'curve_stop', 'is_curve',
                    'remaining', 'global_remaining', 'circuit_seg', 'stop',
                    'circuit_seg_1', 'stop_1',
                    'idx'
                ])
            df.to_csv(self.log_file_name)
