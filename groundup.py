from dice import six_sided, four_sided, make_test_dice
from ucb import main, trace, interact
import numpy as np
import random

GOAL_SCORE = 100  # The goal of Hog is to score 100 points.


def roll_dice(num_rolls, dice=six_sided):
    assert type(num_rolls) == int, 'num_rolls must be an integer.'
    assert num_rolls > 0, 'Must roll at least once.'

    score = 0
    pigout = False
    for i in range(num_rolls):
        roll = dice()
        if roll == 1:
            pigout = True
        score += roll
    if pigout:
        return 1
    return score

def make_averaged(fn, num_samples=1000):
    """Return a function that returns the average value of FN when called.
    To implement this function, you will have to use *args syntax, a new Python
    feature introduced in this project.  See the project description.
    >>> dice = make_test_dice(4, 2, 5, 1)
    >>> averaged_dice = make_averaged(dice, 1000)
    >>> averaged_dice()
    3.0
    """
    # BEGIN PROBLEM 8
    def average(*args):
        value = 0
        for x in range(num_samples):
            value += fn(*args)
        return value / num_samples
    return average
    # END PROBLEM 8

def free_bacon(score):
    print(score)
    assert score < 100, 'The game should be over.'
    if score < 10:
        return 10
    a = score // 10
    b = score % 10
    return 10 - min(a, b)


def take_turn(num_rolls, opponent_score, dice=six_sided):
    assert type(num_rolls) == int, 'num_rolls must be an integer.'
    assert num_rolls >= 0, 'Cannot roll a negative number of dice in take_turn.'
    assert num_rolls <= 10, 'Cannot roll more than 10 dice.'
    assert opponent_score < 100, 'The game should be over.'

    if num_rolls != 0:
        return make_averaged(roll_dice)(num_rolls, dice)
    else:
        return free_bacon(opponent_score)


def swapmult(score):
    if score > 100:
        a = score // 100
        b = score % 10
    elif score > 9:
        a = score // 10
        b = score % 10
    else:
        a = score
        b = score
    return a * b


def is_swap(player_score, opponent_score):
    if swapmult(player_score) == swapmult(opponent_score):
        return True
    else:
        return False

def complete_turn(num_rolls, score0, score1, prev_num_rolls, dice=six_sided):
    new0 = num_rolls
    score0 += take_turn(new0, score1, dice)
    if abs(new0 - prev_num_rolls) == 2:
        score0 += 3
    if is_swap(score0, score1):
        score0, score1 = score1, score0
    prev_num_rolls = new0
    return score0, score1, prev_num_rolls

# V1 Features: score, opponent
# V2 Features: free_bacon(opponent)
# V3 Features: swapmult(opponent)

lr = 0.1
gamma = 0.75
Q = np.zeros((101, 101, 11))
# Need to convert this to a dictionary at the end

def update(score0, score1, action, nextscore0, nextscore1):
    Q[score0, score1, action] = Q[score0, score1, action] + (lr * (1 + gamma * np.max(Q[nextscore0, nextscore1, :]) - Q[score0, score1, action]))

def main(games):
    for i in range(games):
        record0 = []
        record1 = []
        score0, score1, prev0, prev1 = 0, 0, 0, 0
        epsilon = 0.2

        while True:
            if random.uniform(0, 1) < epsilon or score0==0:
                randrolls = random.randint(0,10)
                record0.append([score0,score1,randrolls])
                score0, score1, prev0 = complete_turn(randrolls, score0, score1, prev0)
                score0, score1, prev1 = int(round(score0)), int(round(score1)), int(round(prev0))
                if score0>100: score0=100
                if score1>100: score1=100
            else:
                qrolls = int(np.argmax(Q[score0,score1,:]))
                record0.append([score0, score1, qrolls])
                score0, score1, prev0 = complete_turn(qrolls, score0, score1, prev0)
                score0, score1, prev1 = int(round(score0)), int(round(score1)), int(round(prev0))
                if score0>100: score0=100
                if score1>100: score1=100

            if score0>=100:
                for i in range(len(record0)-1):
                    update(record0[i][0],record0[i][1],record0[i][2], record0[i+1][0], record0[i+1][1])
                    break
            elif score1>=100:
                for i in range(len(record1)-1):
                    update(record1[i][0],record1[i][1],record1[i][2], record1[i+1][0], record1[i+1][1])
                    break

            if random.uniform(0, 1) < epsilon or score1==0:
                randrolls = random.randint(0, 10)
                record1.append([score0, score1, randrolls])
                score0, score1, prev1 = complete_turn(randrolls, score0, score1, prev1)
                score0, score1, prev1 = int(round(score0)), int(round(score1)), int(round(prev1))
                if score0>100: score0=100
                if score1>100: score1=100
            else:
                qrolls = int(np.argmax(Q[score0, score1, :]))
                print(qrolls)
                record1.append([score0, score1, qrolls])
                score0, score1, prev01 = complete_turn(qrolls, score0, score1, prev1)
                score0, score1, prev1 = int(round(score0)), int(round(score1)), int(round(prev1))
                if score0>100: score0=100
                if score1>100: score1=100

            if score0>=100:
                for i in range(len(record0) - 1):
                    update(record0[i][0], record0[i][1], record0[i][2], record0[i + 1][0], record0[i + 1][1])
                    break
            elif score1 >= 100:
                for i in range(len(record1) - 1):
                    update(record1[i][0], record1[i][1], record1[i][2], record1[i + 1][0], record1[i + 1][1])
                    break

main(10)
