#!/usr/bin/env python
import logging
import random
import copy
from namedlist import namedlist
# length, nb_stops, length_curve]
CIRCUIT = [[22, 7, 18, 10, 2, 6, 8, 6, 3, 18, 28, 7, 2, 7, 2, 13, 3, 11, 6, 6],
           [1, 1, 1, 1, 3, 1, 1, 2, 2, 1]]

DICE = [
    [1, 1, 2, 2],
    [2, 3, 3, 4, 4, 4],
    [
        4,
        5,
        6,
        6,
        7,
        7,
        8,
        8,
    ],
    range(7, 13),
    range(11, 21),
    range(21, 31),
]

Candidate = namedlist('Candidate', 'position health gear curve_stop done')


class NoIndex(Exception):
    pass


class Crash(Exception):
    pass


class InvalidGear(Exception):
    pass


def get_index_for_position(position, circuit):
    cur_sum = 0
    for k, v in enumerate(circuit[0]):
        cur_sum += v
        if cur_sum >= position:
            return k
    return len(circuit[0])


def get_position_for_index(index, circuit):
    return sum(circuit[0][:index])


def is_curve(index):
    return index % 2 == 1


def is_done(index, circuit):
    return index == len(circuit[0])


def check_curve_leave(candidate, cur_index, next_position, circuit):
    curve_index = int(cur_index / 2)
    nb_expected_stop = circuit[1][curve_index]
    if nb_expected_stop - candidate.curve_stop >= 2:
        raise Crash()
    if nb_expected_stop - candidate.curve_stop < 1:
        return
    end_turn = sum(circuit[0][:cur_index + 1])
    reduce_health(candidate, next_position - end_turn)


def reduce_health(candidate, health_reduction):
    candidate.health -= health_reduction
    if candidate.health < 0:
        raise Crash()


def update_candidate(candidate, dice_value, circuit):
    ''' '''
    cur_position = candidate.position
    cur_index = get_index_for_position(cur_position, circuit)
    is_cur_curve = is_curve(cur_index)
    next_position = cur_position + dice_value
    next_index = get_index_for_position(next_position, circuit)
    # straight
    if not is_cur_curve:
        # next is same => update
        if cur_index == next_index:
            candidate.position = next_position
            return
        # next is + 1 => update with curve_stop
        if next_index == cur_index + 1:
            candidate.position = next_position
            candidate.curve_stop = 1
            candidate.done = is_done(next_index, circuit)
            return
        # next is + 2 => crash
        raise Crash('missed a curve')
    # curve
    if cur_index == next_index:
        # next is same => update with curve_stop
        candidate.position = next_position
        candidate.curve_stop += 1
        return
    if next_index == cur_index + 1:
        # next is + 1 => check curve_stop , reduce health
        check_curve_leave(candidate, cur_index, next_position, circuit)
        candidate.position = next_position
        candidate.curve_stop = 0
        candidate.done = is_done(next_index, circuit)
        return
    if next_index == cur_index + 2:
        # next is + 2 => check curve_stop, reduce_health, update with curve_stop
        check_curve_leave(candidate, cur_index, next_position, circuit)
        candidate.position = next_position
        candidate.curve_stop = 1
        candidate.done = is_done(next_index, circuit)
        return
    # next is + 3 => crash
    raise Crash('skipped a curve')


def race(evaluation_function, circuit):
    race_log = []
    candidate = Candidate(
        position=0, health=10, gear=0, curve_stop=0, done=False)
    failed = False
    nb_turn = 0
    try:
        while not candidate.done:
            gear_modifier = evaluation_function(candidate, circuit)
            candidate.gear += gear_modifier
            if candidate.gear < 0 or candidate.gear > 6:
                raise InvalidGear()

            move = random.choice(DICE[candidate.gear])
            update_candidate(candidate, move, circuit)
            nb_turn += 1
            race_log.append(copy.copy(candidate))
    except Exception as e:
        logging.exception(e)
        failed = True

    return (score(candidate, nb_turn, failed), race_log)


def score(candidate, nb_turn, failed):
    if failed:
        return (0, candidate.position)
    return (nb_turn, 0)


def log_to_dataframe(race):
    import pandas as pd
    l = [(c.position, c.health, c.gear, c.curve_stop) for c in race]
    df = pd.DataFrame(l, columns=['position', 'health', 'gear', 'curve_stop'])
    return df
