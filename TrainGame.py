from dice import six_sided, four_sided, make_test_dice
from ucb import main, trace, interact
import numpy as np

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


def free_bacon(score):
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
        return roll_dice(num_rolls, dice)
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


def take_turn0(num_rolls, score0, score1, prev_num_rolls, dice=six_sided):
    new0 = num_rolls
    score0 += take_turn(new0, score1, dice)
    if abs(new0 - prev_num_rolls) == 2:
        score0 += 3
    if is_swap(score0, score1):
        score0, score1 = score1, score0
    prev_num_rolls = new0
    return score0, score1, prev_num_rolls


def take_turn1(num_rolls, score0, score1, prev_num_rolls, dice=six_sided):
    new1 = num_rolls
    score1 += take_turn(new1, score0, dice)
    if abs(new1 - prev_num_rolls) == 2:
        score1 += 3
    if is_swap(score0, score1):
        score0, score1 = score1, score0
    prev_num_rolls = new1
    return score0, score1, prev_num_rolls

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

# V1 Features: score, opponent
# V2 Features: free_bacon(opponent)
# V3 Features: swapmult(opponent)

class State:
    def __init__(self, p1, p2):
        self.board = [0, 0]
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = 1

    # get unique hash of current board state
    def getHash(self):
        self.boardHash = ''
        for i in self.board:
            if i < 10:
                self.boardHash += ('00' + str(i))
            elif i < 100:
                self.boardHash += ('0' + str(i))
            else:
                self.boardHash += (str(i))
        return self.boardHash

    def winner(self):
        if self.board[0] >= 100:
            return 1
        elif self.board[1] >= 100:
            return -1

    def availablePositions(self):
        return [i for i in range(11)]

    def updateState(self, num_rolls):
        if self.playerSymbol == 1:
            self.board[0], self.board[1], self.p1.prev = take_turn0(num_rolls, self.board[0], self.board[1], self.p1.prev)
        else:
            self.board[0], self.board[1], self.p2.prev = take_turn1(num_rolls, self.board[0], self.board[1], self.p2.prev)
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)  # Why are these nonzero, should be 0 since no one won?
            self.p2.feedReward(0.5)  # Why are these nonzero, should be 0 since no one won?

    # board reset
    def reset(self):
        self.board = [0, 0]
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=100):
        for i in range(rounds):
            if i % 1000 == 0:
                print("Rounds {}".format(i))
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self)
                # take action and upate board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end

                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break


class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value
        self.prev = 0

    def getHash(self, board):
        boardHash = ''
        for i in board:
            if i < 10:
                boardHash += ('00' + str(i))
            elif i < 100:
                boardHash += ('0' + str(i))
            else:
                boardHash += (str(i))
        return boardHash

    def get_value(self, num_rolls, board):
        next_board = board.deepcopy()
        next_board.updateState(num_rolls)
        next_boardHash = self.getHash(next_board)
        value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
        return value

    def chooseAction(self, positions, board):
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                if p == 0:
                    value = self.get_value(p, board)
                else:
                    averaged_value = make_averaged(self.get_value, num_samples=1000)
                    value = averaged_value(p)
                if value >= value_max:
                    value_max = value
                    action = p
        print("{} takes action {}".format(self.name, action))
        return action

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


if __name__ == "__main__":
    # training
    p1 = Player("p1")
    p2 = Player("p2")

    st = State(p1, p2)
    print("training...")
    st.play(50000)
